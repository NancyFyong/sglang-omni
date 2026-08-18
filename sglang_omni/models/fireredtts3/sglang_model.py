# SPDX-License-Identifier: Apache-2.0
"""SGLang Qwen3 backbone with the FireRedTTS3 continuous-latent head."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from torch import nn

from sglang_omni.models.fireredtts3.flow_head import FireRedTTS3LatentHead
from sglang_omni.models.weight_loader import default_weight_loader

_HEAD_PREFIXES = (
    "spk_proj_llm.",
    "spk_proj_dit.",
    "patch_encoder.",
    "dit_head.",
    "dit.",
    "stop_head.",
)


class FireRedTTS3SGLangModel(nn.Module):
    """Keep AR execution in SGLang; add only FireRedTTS3-specific operators."""

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.backbone = Qwen3ForCausalLM(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.backbone" if prefix else "backbone",
        )
        self.head = FireRedTTS3LatentHead(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        head_params = dict(self.head.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith("backbone_llm."):
                backbone_weights.append((name.removeprefix("backbone_llm."), tensor))
                continue
            if not name.startswith(_HEAD_PREFIXES):
                raise AssertionError(
                    f"Unexpected FireRedTTS3 checkpoint weight {name!r}; expected a "
                    "backbone_llm weight or a latent-head parameter"
                )
            parameter = head_params.get(name)
            if parameter is None:
                raise AssertionError(
                    f"FireRedTTS3 latent head has no parameter for {name!r}"
                )
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(f"head.{name}")
        self.backbone.load_weights(backbone_weights)
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        del kwargs
        if input_embeds is None:
            input_embeds = forward_batch.input_embeds
        if input_embeds is None:
            raise RuntimeError(
                "FireRedTTS3 decodes from continuous embeddings; the model runner "
                "must stage input_embeds for every forward"
            )
        hidden_states = self.backbone.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        # FireRedTTS3 consumes hidden states only and the runner overwrites
        # next_token_ids, so dummy logits just satisfy the SGLang contract.
        if forward_batch.forward_mode.is_extend():
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            request_count = (
                int(extend_seq_lens.numel()) if extend_seq_lens is not None else 1
            )
        else:
            request_count = int(hidden_states.shape[0])
        return LogitsProcessorOutput(
            next_token_logits=hidden_states.new_empty((request_count, 1)),
            hidden_states=hidden_states,
        )


EntryClass = FireRedTTS3SGLangModel

__all__ = ["FireRedTTS3SGLangModel", "EntryClass"]
