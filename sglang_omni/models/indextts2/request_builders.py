# SPDX-License-Identifier: Apache-2.0
"""Map IndexTTS-2.5 pipeline state onto SGLang's native request lifecycle.

Unlike the continuous-latent TTS families, IndexTTS-2.5 emits discrete mel
codes, so SGLang owns sampling and stop handling; only the input embeddings are
model-specific.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.indextts2.payload_types import (
    IndexTTS2State,
    load_indextts2_state,
    store_indextts2_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

#: Prefix rows whose embedding is spliced in by the runner still need a token id
#: for SGLang's KV bookkeeping. Upstream fills those positions with 1, and its
#: repetition penalty sees them, so the same id is used here.
PLACEHOLDER_TOKEN_ID = 1


@dataclass
class IndexTTS2SGLangRequestData(SGLangARRequestData):
    state: IndexTTS2State = field(default_factory=IndexTTS2State)
    prefix_length: int = 0
    engine_start_s: float = 0.0


def build_sglang_indextts2_request(
    payload: StagePayload,
    *,
    config: Any,
) -> IndexTTS2SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = load_indextts2_state(payload)
    text_tokens = state.text_token_ids
    if text_tokens is None or text_tokens.numel() == 0:
        raise RuntimeError("IndexTTS-2.5 preprocessing did not emit text tokens")
    if state.speaker_embedding is None or state.emotion_embedding is None:
        raise RuntimeError("IndexTTS-2.5 reference encoding did not run")
    if state.prompt_condition is None or state.reference_mel is None:
        raise RuntimeError("IndexTTS-2.5 reference encoding produced no s2mel prompt")

    start_text = int(config.start_text_token)
    stop_text = int(config.stop_text_token)
    ids = text_tokens.reshape(-1).tolist()
    text_length = sum(1 for value in ids if value not in (start_text, stop_text))
    if text_length == 0:
        raise ValueError("IndexTTS-2.5 requires at least one text token")
    if text_length + 2 > int(config.max_text_tokens) + 2:
        raise ValueError(
            f"IndexTTS-2.5 accepts at most {config.max_text_tokens} text tokens, "
            f"got {text_length}"
        )
    # [speaker+emotion, reserved, reserved] + [start] text [stop] + [start_mel]
    prefix_length = int(config.condition_tokens) + text_length + 2
    max_new_tokens = int(state.max_mel_tokens)
    if max_new_tokens <= 0 or max_new_tokens > int(config.max_mel_tokens):
        raise ValueError(
            f"IndexTTS-2.5 max_mel_tokens must be in [1, {config.max_mel_tokens}], "
            f"got {max_new_tokens}"
        )
    if prefix_length + 1 + max_new_tokens > int(config.n_positions):
        raise ValueError(
            f"IndexTTS-2.5 request needs {prefix_length + 1 + max_new_tokens} "
            f"positions but the context length is {config.n_positions}"
        )

    input_ids = [PLACEHOLDER_TOKEN_ID] * prefix_length + [int(config.start_mel_token)]
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=float(state.temperature),
        top_p=float(state.top_p),
        top_k=int(state.top_k),
        # The CTRL-style penalty is applied by the model (see
        # IndexTTS2SGLangModel.stage_repetition_penalty) because SGLang caps its
        # native repetition_penalty at 2.0 and this model ships 10.0.
        repetition_penalty=1.0,
        stop_token_ids=[int(config.stop_mel_token)],
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(config.number_mel_codes))
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        eos_token_ids={int(config.stop_mel_token)},
        vocab_size=int(config.number_mel_codes),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None
    return IndexTTS2SGLangRequestData(
        state=state,
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        input_ids=torch.tensor([input_ids], dtype=torch.long),
        prefix_length=prefix_length,
        max_new_tokens=max_new_tokens,
        temperature=float(state.temperature),
        top_p=float(state.top_p),
        top_k=int(state.top_k),
        repetition_penalty=float(state.repetition_penalty),
        input_embeds_are_projected=True,
        engine_start_s=time.perf_counter(),
    )


def apply_mel_code_result(
    data: IndexTTS2SGLangRequestData, *, stop_mel_token: int
) -> StagePayload:
    state = data.state
    codes = [int(value) for value in data.output_ids]
    if stop_mel_token in codes:
        codes = codes[: codes.index(stop_mel_token)]
    if not codes:
        raise RuntimeError("IndexTTS-2.5 generated no mel codes")
    state.mel_codes = torch.tensor([codes], dtype=torch.long)
    state.text_token_ids = None
    state.prompt_tokens = int(data.input_ids.numel())
    state.completion_tokens = len(codes)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    state.finish_reason = data.finish_reason
    return store_indextts2_state(data.stage_payload, state)


__all__ = [
    "PLACEHOLDER_TOKEN_ID",
    "IndexTTS2SGLangRequestData",
    "apply_mel_code_result",
    "build_sglang_indextts2_request",
]
