# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 latent head: patch encoder, DiT flow matching, and stop head.

The backbone runs inside SGLang; everything that turns a backbone hidden state
into the next latent patch (and the continuous embedding fed back into the next
decode step) lives here. Upstream keeps these operators in float32 while the
Qwen3 backbone runs under bf16 autocast, so this module pins float32 too.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sglang_omni.models.fireredtts3.components.dit import DiT
from sglang_omni.models.fireredtts3.components.patch_encoder import PatchEncoder
from sglang_omni.models.fireredtts3.hf_config import FireRedTTS3Config

_HEAD_DTYPE = torch.float32


@dataclass
class FireRedFlowState:
    """Per-request recurrence carried across AR steps."""

    latents: torch.Tensor
    """(1, history_length + prompt + generated, redae_dim); leading zeros are the dummy history."""
    cond_history: torch.Tensor
    """(1, >=num_history_patches, hidden_size) backbone rows, one per emitted patch."""
    speaker: torch.Tensor
    """(1, spk_in_dim) raw CAM++ speaker embedding."""
    spk_cond: torch.Tensor
    """(1, spk_in_dim) speaker condition for the DiT."""
    n_timesteps: int
    inference_cfg: float
    generator: torch.Generator | None = None
    prompt_frames: int = 0
    patch_count: int = 0


@dataclass
class FireRedFlowStep:
    """One decoded latent patch plus the embedding fed into the next step."""

    latent_patch: torch.Tensor
    feedback_embedding: torch.Tensor


class FireRedTTS3LatentHead(nn.Module):
    """Patch encoder + DiT flow head + stop head + speaker projections."""

    def __init__(self, config: FireRedTTS3Config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.redae_dim = int(config.redae_dim)
        self.patch_size = int(config.patch_size)
        self.history_patches = int(config.num_history_patches)
        self.history_length = self.history_patches * self.patch_size
        self.hidden_size = hidden_size
        self.spk_in_dim = int(config.spk_in_dim)

        self.spk_proj_llm = nn.Linear(self.spk_in_dim, hidden_size)
        self.spk_proj_dit = nn.Linear(self.spk_in_dim, self.spk_in_dim)
        self.patch_encoder = PatchEncoder(
            in_dim=self.redae_dim,
            out_dim=hidden_size,
            patch_size=self.patch_size,
            hidden_size=int(config.patch_encoder_hidden_size),
            mlp_ratio=int(config.patch_encoder_mlp_ratio),
            depth=int(config.patch_encoder_depth),
            num_heads=int(config.patch_encoder_num_heads),
        )
        self.dit_head = nn.Linear(hidden_size, int(config.dit_hidden_size))
        self.dit = DiT(
            in_channels=config.dit_in_channels,
            out_channels=self.redae_dim,
            mlp_ratio=int(config.dit_mlp_ratio),
            depth=int(config.dit_depth),
            num_heads=int(config.dit_num_heads),
            hidden_size=int(config.dit_hidden_size),
        )
        self.stop_head = nn.Linear(hidden_size, 1)
        self.to(_HEAD_DTYPE)
        self._t_span_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # request lifecycle
    # ------------------------------------------------------------------ #
    def new_request(
        self,
        *,
        prompt_latents: torch.Tensor,
        speaker_embedding: torch.Tensor,
        n_timesteps: int,
        inference_cfg: float,
        seed: int | None,
    ) -> FireRedFlowState:
        """Build the per-request recurrence state from the encoded reference."""
        device = self.device
        latents = prompt_latents.to(device=device, dtype=_HEAD_DTYPE)
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                f"FireRedTTS3 prompt latents must be (1, frames, {self.redae_dim}), "
                f"got {tuple(latents.shape)}"
            )
        if latents.shape[1] == 0 or latents.shape[1] % self.patch_size:
            raise ValueError(
                "FireRedTTS3 prompt latents must be a non-empty multiple of "
                f"patch_size={self.patch_size}, got {latents.shape[1]} frames"
            )
        speaker = speaker_embedding.to(device=device, dtype=_HEAD_DTYPE)
        if speaker.ndim == 1:
            speaker = speaker.unsqueeze(0)
        if speaker.shape != (1, self.spk_in_dim):
            raise ValueError(
                f"FireRedTTS3 speaker embedding must be (1, {self.spk_in_dim}), "
                f"got {tuple(speaker.shape)}"
            )
        if int(n_timesteps) <= 0:
            raise ValueError("FireRedTTS3 n_timesteps must be positive")

        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        history = latents.new_zeros((1, self.history_length, self.redae_dim))
        state = FireRedFlowState(
            latents=torch.cat([history, latents], dim=1),
            cond_history=latents.new_zeros((1, self.history_patches, self.hidden_size)),
            speaker=speaker,
            spk_cond=self.spk_proj_dit(speaker),
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            generator=generator,
            prompt_frames=int(latents.shape[1]),
        )
        return state

    def prefill_embeddings(
        self, state: FireRedFlowState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(speaker_row, prompt_patch_embeds)`` for a (re-)prefill.

        ``speaker_row`` is the leading prefill row and ``prompt_patch_embeds``
        are the patchified reference latents appended after the text tokens.
        """
        prompt = state.latents[
            :, self.history_length : self.history_length + state.prompt_frames
        ]
        return self.spk_proj_llm(state.speaker), self.patch_encoder(prompt)[0]

    def release_request(self, state: FireRedFlowState) -> None:
        state.latents = state.latents[:, :0]
        state.cond_history = state.cond_history[:, :0]

    # ------------------------------------------------------------------ #
    # per-step operators
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return self.stop_head.weight.device

    def encode_patches(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify ``(B, patch_size, redae_dim)`` latents into ``(B, hidden)``."""
        embeds = self.patch_encoder(latents.to(dtype=_HEAD_DTYPE))
        return embeds.reshape(-1, self.hidden_size)

    def stop_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Sigmoid stop probability for ``(B, hidden_size)`` backbone rows."""
        logits = self.stop_head(hidden.to(dtype=_HEAD_DTYPE)).squeeze(-1)
        return torch.sigmoid(logits)

    def initialize_history(
        self, state: FireRedFlowState, patch_hidden: torch.Tensor
    ) -> None:
        """Rebuild the condition history from prefill rows.

        ``patch_hidden`` holds one row per already-emitted latent patch, i.e.
        the reference patches plus any patch replayed after a retraction. The
        leading zero rows are upstream's dummy history and are re-created here
        so a re-prefill lands on the same state as the first prefill.
        """
        rows = patch_hidden.to(device=self.device, dtype=_HEAD_DTYPE)
        if rows.ndim == 2:
            rows = rows.unsqueeze(0)
        zeros = rows.new_zeros((1, self.history_patches, self.hidden_size))
        state.cond_history = torch.cat([zeros, rows], dim=1)

    def decode_batch(
        self,
        states: list[FireRedFlowState],
        hidden: torch.Tensor,
        *,
        append_hidden: bool,
    ) -> list[FireRedFlowStep]:
        """Advance one latent patch for every request in the batch."""
        if len(states) != hidden.shape[0]:
            raise RuntimeError(
                f"FireRedTTS3 flow batch mismatch: {len(states)} states vs "
                f"{hidden.shape[0]} hidden rows"
            )
        rows = hidden.to(device=self.device, dtype=_HEAD_DTYPE)
        if append_hidden:
            for state, row in zip(states, rows, strict=True):
                state.cond_history = torch.cat(
                    [state.cond_history, row.view(1, 1, -1)], dim=1
                )

        patches: list[torch.Tensor | None] = [None] * len(states)
        groups: dict[int, list[int]] = {}
        for index, state in enumerate(states):
            groups.setdefault(state.n_timesteps, []).append(index)
        for n_timesteps, indices in groups.items():
            group_states = [states[index] for index in indices]
            decoded = self._flow_group(group_states, n_timesteps)
            for offset, index in enumerate(indices):
                patches[index] = decoded[offset : offset + 1]

        steps: list[FireRedFlowStep] = []
        stacked = torch.cat([patch for patch in patches], dim=0)
        feedback = self.encode_patches(stacked)
        for index, state in enumerate(states):
            patch = patches[index]
            state.latents = torch.cat([state.latents, patch], dim=1)
            state.patch_count += 1
            steps.append(
                FireRedFlowStep(
                    latent_patch=patch,
                    feedback_embedding=feedback[index],
                )
            )
        return steps

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _t_span(self, n_timesteps: int) -> torch.Tensor:
        key = (int(n_timesteps), self.device)
        cached = self._t_span_cache.get(key)
        if cached is None:
            span = torch.linspace(
                0, 1, int(n_timesteps) + 1, device=self.device, dtype=_HEAD_DTYPE
            )
            cached = 1 - torch.cos(span * 0.5 * torch.pi)
            self._t_span_cache[key] = cached
        return cached

    def _flow_group(
        self, states: list[FireRedFlowState], n_timesteps: int
    ) -> torch.Tensor:
        """Batched flow-matching ODE for one ``n_timesteps`` group."""
        batch = len(states)
        window = self.history_patches + 1
        hist = torch.cat(
            [state.latents[:, -self.history_length :] for state in states], dim=0
        )
        cond_rows = torch.cat(
            [state.cond_history[:, -window:] for state in states], dim=0
        )
        spk = torch.cat([state.spk_cond for state in states], dim=0)

        x0 = torch.stack(
            [
                torch.randn(
                    (self.patch_size, self.redae_dim),
                    device=self.device,
                    dtype=_HEAD_DTYPE,
                    generator=state.generator,
                )
                for state in states
            ],
            dim=0,
        )
        xt = torch.cat([hist, x0], dim=1)
        span_len = self.history_length + self.patch_size
        cond = torch.cat(
            [
                self.dit_head(cond_rows).repeat_interleave(self.patch_size, dim=1),
                spk.unsqueeze(1).expand(batch, span_len, self.spk_in_dim),
            ],
            dim=-1,
        )
        cfg = torch.tensor(
            [state.inference_cfg for state in states],
            device=self.device,
            dtype=_HEAD_DTYPE,
        ).view(batch, 1, 1)
        use_cfg = bool((cfg > 0).any().item())

        t_span = self._t_span(n_timesteps)
        for step in range(t_span.shape[0] - 1):
            t = t_span[step]
            dt = t_span[step + 1] - t
            x_in = torch.cat([xt, cond], dim=2)
            if use_cfg:
                x_in = torch.cat([x_in, torch.cat([xt, cond * 0], dim=2)], dim=0)
            # t is a scalar per ODE step; every row (including the CFG rows)
            # shares it, so the shape is derived from the assembled batch.
            t_in = t.view(1, 1, 1).expand(x_in.shape[0], 1, 1)
            vt = self.dit(x=x_in, t=t_in)
            if use_cfg:
                vt_cond, vt_uncond = vt.chunk(2, dim=0)
                vt = (1.0 + cfg) * vt_cond - cfg * vt_uncond
            xt[:, -self.patch_size :] = (
                xt[:, -self.patch_size :] + dt.view(1, 1, 1) * vt[:, -self.patch_size :]
            )
        return xt[:, -self.patch_size :]


__all__ = [
    "FireRedFlowState",
    "FireRedFlowStep",
    "FireRedTTS3LatentHead",
]
