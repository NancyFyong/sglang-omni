# SPDX-License-Identifier: Apache-2.0
"""Per-request state carried across the FireRedTTS3 pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class FireRedTTS3State(DeclarativeStateBase):
    """Only cross-stage state; scheduler/runner state stays in Omni."""

    sample_rate: int = wire(24000, codec="int")
    # ---- preprocessing output ----
    text_token_ids: torch.Tensor | None = wire(None, codec="typed_tensor")
    language: str = wire("Chinese", codec="str_or")
    reference_audio: str | None = None
    # ---- reference encode output ----
    prompt_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    speaker_embedding: torch.Tensor | None = wire(None, codec="typed_tensor")
    # ---- engine sampling knobs ----
    n_timesteps: int = wire(10, codec="int_or")
    inference_cfg: float = wire(2.0, codec="float")
    stop_threshold: float = wire(0.5, codec="float")
    min_gen_steps: int = wire(6, codec="int")
    max_gen_steps: int = wire(400, codec="int_or")
    seed: int | None = wire(None, codec="opt_int")
    # ---- engine output ----
    generated_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    finish_reason: str | None = None


def load_fireredtts3_state(payload: StagePayload) -> FireRedTTS3State:
    return FireRedTTS3State.from_dict(payload.data)


def store_fireredtts3_state(
    payload: StagePayload, state: FireRedTTS3State
) -> StagePayload:
    payload.data = state.to_dict()
    return payload


__all__ = [
    "FireRedTTS3State",
    "load_fireredtts3_state",
    "store_fireredtts3_state",
]
