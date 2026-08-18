# SPDX-License-Identifier: Apache-2.0
"""Map FireRedTTS3 pipeline state onto SGLang's native request lifecycle."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from sglang_omni.models.fireredtts3.payload_types import (
    FireRedTTS3State,
    load_fireredtts3_state,
    store_fireredtts3_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

if TYPE_CHECKING:
    from sglang_omni.models.fireredtts3.flow_head import FireRedFlowState

#: Prefill rows whose embedding is spliced in by the runner still need a token id
#: for SGLang's KV bookkeeping; BOS is inert here because the ids are never read.
PLACEHOLDER_TOKEN_ID = 151643


@dataclass
class FireRedTTS3SGLangRequestData(SGLangARRequestData):
    state: FireRedTTS3State = field(default_factory=FireRedTTS3State)
    prompt_patch_count: int = 0
    text_token_count: int = 0
    flow_state: "FireRedFlowState | None" = None
    latent_patches: list[torch.Tensor] = field(default_factory=list)
    control_token_id: int = PLACEHOLDER_TOKEN_ID
    engine_start_s: float = 0.0


def build_sglang_fireredtts3_request(
    payload: StagePayload,
    *,
    patch_size: int,
    max_sequence_length: int,
) -> FireRedTTS3SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = load_fireredtts3_state(payload)
    text_tokens = state.text_token_ids
    if text_tokens is None or text_tokens.numel() == 0:
        raise RuntimeError("FireRedTTS3 preprocessing did not emit text tokens")
    if state.prompt_latents is None or state.speaker_embedding is None:
        raise RuntimeError("FireRedTTS3 reference encoding did not run")
    prompt_frames = int(state.prompt_latents.shape[-2])
    if prompt_frames <= 0 or prompt_frames % patch_size:
        raise ValueError(
            f"FireRedTTS3 prompt latents must be a multiple of {patch_size} frames, "
            f"got {prompt_frames}"
        )
    prompt_patch_count = prompt_frames // patch_size

    text_ids = text_tokens.reshape(-1).to(dtype=torch.long).tolist()
    # [speaker row] + text tokens + [one row per reference latent patch]
    input_ids = (
        [PLACEHOLDER_TOKEN_ID] + text_ids + [PLACEHOLDER_TOKEN_ID] * prompt_patch_count
    )
    max_gen_steps = int(state.max_gen_steps)
    if max_gen_steps <= 0:
        raise ValueError("FireRedTTS3 max_gen_steps must be positive")
    if len(input_ids) + max_gen_steps > int(max_sequence_length):
        raise ValueError(
            f"FireRedTTS3 request needs {len(input_ids) + max_gen_steps} positions "
            f"but the engine context length is {max_sequence_length}"
        )

    sampling_params = SamplingParams(
        max_new_tokens=max_gen_steps,
        temperature=0.0,
        stop_token_ids=[],
    )
    sampling_params.normalize(None)
    sampling_params.verify(PLACEHOLDER_TOKEN_ID + 1)
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        eos_token_ids=set(),
        vocab_size=PLACEHOLDER_TOKEN_ID + 1,
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None
    return FireRedTTS3SGLangRequestData(
        state=state,
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        input_ids=torch.tensor([input_ids], dtype=torch.long),
        prompt_patch_count=prompt_patch_count,
        text_token_count=len(text_ids),
        max_new_tokens=max_gen_steps,
        input_embeds_are_projected=True,
        engine_start_s=time.perf_counter(),
    )


def apply_latent_result(data: FireRedTTS3SGLangRequestData) -> StagePayload:
    state = data.state
    if not data.latent_patches:
        raise RuntimeError("FireRedTTS3 generated no latent patches")
    state.generated_latents = torch.cat(data.latent_patches, dim=1)
    state.text_token_ids = None
    state.prompt_tokens = int(data.input_ids.numel())
    state.completion_tokens = len(data.latent_patches)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    state.finish_reason = data.finish_reason
    return store_fireredtts3_state(data.stage_payload, state)


__all__ = [
    "PLACEHOLDER_TOKEN_ID",
    "FireRedTTS3SGLangRequestData",
    "apply_latent_result",
    "build_sglang_fireredtts3_request",
]
