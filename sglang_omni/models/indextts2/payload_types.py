# SPDX-License-Identifier: Apache-2.0
"""Per-request state carried across the IndexTTS-2.5 pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class IndexTTS2State(DeclarativeStateBase):
    """Only cross-stage state; scheduler/runner state stays in Omni."""

    sample_rate: int = wire(22050, codec="int")
    # ---- preprocessing output ----
    text_token_ids: torch.Tensor | None = wire(None, codec="typed_tensor")
    language: str = wire("zh", codec="str_or")
    language_id: int = wire(0, codec="int")
    reference_audio: str | None = None
    emotion_audio: str | None = None
    emotion_vector: list[float] | None = wire(None, codec="list")
    emotion_alpha: float = wire(1.0, codec="float")
    emotion_random: bool = wire(False, codec="bool")
    # ---- reference encode output ----
    speaker_embedding: torch.Tensor | None = wire(None, codec="typed_tensor")
    emotion_embedding: torch.Tensor | None = wire(None, codec="typed_tensor")
    prompt_condition: torch.Tensor | None = wire(None, codec="typed_tensor")
    reference_mel: torch.Tensor | None = wire(None, codec="typed_tensor")
    # ---- sampling knobs ----
    temperature: float = wire(0.8, codec="float")
    top_p: float = wire(0.8, codec="float")
    top_k: int = wire(30, codec="int")
    repetition_penalty: float = wire(10.0, codec="float")
    max_mel_tokens: int = wire(1500, codec="int_or")
    seed: int | None = wire(None, codec="opt_int")
    # ---- vocoder knobs ----
    duration_factor: float = wire(1.0, codec="float")
    diffusion_steps: int = wire(25, codec="int_or")
    inference_cfg_rate: float = wire(0.7, codec="float")
    # ---- engine output ----
    mel_codes: torch.Tensor | None = wire(None, codec="typed_tensor")
    finish_reason: str | None = None


def load_indextts2_state(payload: StagePayload) -> IndexTTS2State:
    return IndexTTS2State.from_dict(payload.data)


def store_indextts2_state(payload: StagePayload, state: IndexTTS2State) -> StagePayload:
    payload.data = state.to_dict()
    return payload


__all__ = [
    "IndexTTS2State",
    "load_indextts2_state",
    "store_indextts2_state",
]
