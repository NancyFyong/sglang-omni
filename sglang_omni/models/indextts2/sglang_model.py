# SPDX-License-Identifier: Apache-2.0
"""SGLang GPT2 backbone with the IndexTTS-2.5 conditioning prefix.

Upstream (``indextts.gpt.model_v2.UnifiedVoice``) deletes GPT2's own token and
position embeddings and drives the transformer purely from spliced embeddings:

* prefix: ``[spk_emb_proj(campplus) + emovec, 0, 0]`` then
  ``text_embedding + text_pos_embedding + lang_embedding``
* mel stream: ``mel_embedding(code) + mel_pos_embedding(index)``

``wpe`` is therefore kept at zero here (upstream replaces it with
``null_position_embeddings``), and the runner supplies ``input_embeds`` for
every forward.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.gpt2 import GPT2Block
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.indextts2.hf_config import IndexTTS2Config
from sglang_omni.models.weight_loader import default_weight_loader

# HF GPT2 stores these as Conv1D, so the checkpoint holds them transposed.
_CONV1D_WEIGHTS = ("c_attn", "c_proj", "c_fc")


class IndexTTS2SGLangModel(nn.Module):
    """GPT2 stack plus the IndexTTS-2.5 embeddings, conditioner, and mel head."""

    def __init__(
        self, config: IndexTTS2Config, quant_config: Any = None, prefix: str = ""
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = int(config.hidden_size)
        self.hidden_size = hidden_size
        self.start_mel_token = int(config.start_mel_token)
        self.stop_mel_token = int(config.stop_mel_token)
        self.start_text_token = int(config.start_text_token)
        self.stop_text_token = int(config.stop_text_token)
        self.number_mel_codes = int(config.number_mel_codes)

        self.h = nn.ModuleList(
            [
                GPT2Block(
                    layer_id,
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"h.{layer_id}", prefix),
                )
                for layer_id in range(int(config.num_hidden_layers))
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.spk_emb_proj = nn.Linear(192, hidden_size)
        self.text_embedding = nn.Embedding(
            int(config.number_text_tokens) + 1, hidden_size
        )
        self.lang_embedding = nn.Embedding(_lang_vocab_size(), hidden_size)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, hidden_size)
        self.text_pos_embedding = nn.Embedding(
            int(config.max_text_tokens) + 2, hidden_size
        )
        self.mel_pos_embedding = nn.Embedding(config.max_mel_positions, hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.mel_head = nn.Linear(hidden_size, self.number_mel_codes)
        self._penalty_mask: torch.Tensor | None = None
        self._penalties: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # embedding assembly
    # ------------------------------------------------------------------ #
    def condition_rows(
        self, speaker_embedding: torch.Tensor, emotion_vector: torch.Tensor
    ) -> torch.Tensor:
        """``[spk_emb_proj(style) + emovec, 0, 0]`` as ``(3, hidden)``."""
        dtype = self.spk_emb_proj.weight.dtype
        device = self.spk_emb_proj.weight.device
        style = speaker_embedding.to(device=device, dtype=dtype).reshape(1, -1)
        emotion = emotion_vector.to(device=device, dtype=dtype).reshape(1, -1)
        conditioned = self.spk_emb_proj(style) + emotion
        reserved = conditioned.new_zeros((2, self.hidden_size))
        return torch.cat([conditioned, reserved], dim=0)

    def text_rows(self, text_token_ids: torch.Tensor, language_id: int) -> torch.Tensor:
        """``text_embedding + text_pos_embedding + lang_embedding`` for one request.

        The ids are re-wrapped with the start/stop text tokens exactly as
        ``UnifiedVoice.prepare_gpt_inputs`` does.
        """
        device = self.text_embedding.weight.device
        ids = text_token_ids.reshape(-1).to(device=device, dtype=torch.long)
        keep = (ids != self.stop_text_token) & (ids != self.start_text_token)
        ids = ids[keep]
        wrapped = torch.cat(
            [
                ids.new_tensor([self.start_text_token]),
                ids,
                ids.new_tensor([self.stop_text_token]),
            ]
        )
        positions = torch.arange(wrapped.numel(), device=device)
        rows = self.text_embedding(wrapped) + self.text_pos_embedding(positions)
        language = torch.tensor([int(language_id)], device=device, dtype=torch.long)
        return rows + self.lang_embedding(language)

    def mel_rows(
        self, mel_token_ids: torch.Tensor, position_indices: torch.Tensor
    ) -> torch.Tensor:
        """``mel_embedding(code) + mel_pos_embedding(index)``."""
        device = self.mel_embedding.weight.device
        ids = mel_token_ids.reshape(-1).to(device=device, dtype=torch.long)
        indices = position_indices.reshape(-1).to(device=device, dtype=torch.long)
        return self.mel_embedding(ids) + self.mel_pos_embedding(indices)

    def mel_position_index(self, mel_offset: int) -> int:
        """Mel position embedding index for stream offset ``mel_offset``.

        Upstream reads the index off the attention-mask width, which makes the
        start token use index 0 and the first generated code use index 2; index
        1 is never used. Reproduced here because the weights were trained with
        it.
        """
        return 0 if mel_offset == 0 else mel_offset + 1

    # ------------------------------------------------------------------ #
    # repetition penalty
    # ------------------------------------------------------------------ #
    def stage_repetition_penalty(
        self, token_ids: list[list[int]], penalties: list[float]
    ) -> None:
        """Stage the CTRL-style penalty context for the next forward.

        SGLang caps its native ``repetition_penalty`` at 2.0 while IndexTTS-2.5
        ships 10.0, so the penalty is applied here instead and the sampler runs
        with 1.0. The penalised set matches upstream: the placeholder prefix id,
        the mel start token, and every code sampled so far.
        """
        if not token_ids:
            self._penalty_mask = None
            self._penalties = None
            return
        device = self.mel_head.weight.device
        mask = torch.zeros(
            (len(token_ids), self.number_mel_codes), dtype=torch.bool, device=device
        )
        for row, ids in enumerate(token_ids):
            if ids:
                mask[row, torch.tensor(ids, dtype=torch.long, device=device)] = True
        self._penalty_mask = mask
        self._penalties = torch.tensor(
            penalties, dtype=torch.float32, device=device
        ).unsqueeze(1)

    def _apply_repetition_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        mask, penalties = self._penalty_mask, self._penalties
        self._penalty_mask = None
        self._penalties = None
        if mask is None or penalties is None:
            return logits
        if mask.shape[0] != logits.shape[0]:
            raise RuntimeError(
                f"IndexTTS-2.5 staged {mask.shape[0]} penalty rows for "
                f"{logits.shape[0]} logit rows"
            )
        penalised = torch.where(logits < 0, logits * penalties, logits / penalties)
        return torch.where(mask, penalised, logits)

    # ------------------------------------------------------------------ #
    # SGLang contract
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        del positions, kwargs
        if input_embeds is None:
            input_embeds = forward_batch.input_embeds
        if input_embeds is None:
            raise RuntimeError(
                "IndexTTS-2.5 decodes from spliced embeddings; the model runner "
                "must stage input_embeds for every forward"
            )
        hidden_states = input_embeds
        for layer in self.h:
            hidden_states = layer(hidden_states, forward_batch)
        hidden_states = self.ln_f(hidden_states)
        rows = self._last_row_indices(input_ids, forward_batch, hidden_states)
        logits = self.mel_head(self.final_norm(hidden_states[rows])).float()
        logits = self._apply_repetition_penalty(logits)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    @staticmethod
    def _last_row_indices(
        input_ids: torch.Tensor, forward_batch: Any, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Rows SGLang samples from: the last position of every sequence."""
        if not forward_batch.forward_mode.is_extend():
            return torch.arange(hidden_states.shape[0], device=hidden_states.device)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return torch.tensor(
                [hidden_states.shape[0] - 1], device=hidden_states.device
            )
        return torch.cumsum(extend_seq_lens.to(hidden_states.device), dim=0) - 1

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, tensor in weights:
            target = _map_weight_name(name)
            if target is None:
                continue
            parameter = params.get(target)
            if parameter is None:
                raise AssertionError(
                    f"Unexpected IndexTTS-2.5 checkpoint weight {name!r} "
                    f"(mapped to {target!r})"
                )
            if name.endswith(".weight") and any(
                marker in name for marker in _CONV1D_WEIGHTS
            ):
                tensor = tensor.t()
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(target)
        missing = sorted(set(params) - loaded)
        if missing:
            raise AssertionError(f"IndexTTS-2.5 weights missing for {missing}")
        return loaded


def _lang_vocab_size() -> int:
    from indextts.utils.tokenizer import LANGUAGE_DICT

    return len(LANGUAGE_DICT) + 1


def _map_weight_name(name: str) -> str | None:
    """Map a ``gpt.pth`` key onto this module, or None when unused here.

    The emotion conditioner (``emo_*``) is loaded by the reference-encode stage
    and ``text_head`` is a training-only head.
    """
    if name.startswith(("emo_conditioning_encoder.", "emo_perceiver_encoder.")):
        return None
    if name.startswith(("emo_layer.", "emovec_layer.", "text_head.")):
        return None
    if name.startswith("gpt."):
        suffix = name.removeprefix("gpt.")
        if suffix.endswith((".attn.bias", ".attn.masked_bias")):
            # Causal-mask buffers; SGLang uses RadixAttention instead.
            return None
        if suffix.startswith(("h.", "ln_f.")):
            return suffix
        # The deleted wte/wpe leave no parameters behind.
        return None
    if name.startswith(("mel_pos_embedding.", "text_pos_embedding.")):
        # LearnedPositionEmbeddings wraps its table in an ``emb`` submodule.
        return name.replace(".emb.weight", ".weight")
    if name.startswith(
        (
            "spk_emb_proj.",
            "text_embedding.",
            "lang_embedding.",
            "mel_embedding.",
            "final_norm.",
            "mel_head.",
        )
    ):
        return name
    raise AssertionError(f"Unknown IndexTTS-2.5 checkpoint weight {name!r}")


EntryClass = IndexTTS2SGLangModel

__all__ = ["IndexTTS2SGLangModel", "EntryClass"]
