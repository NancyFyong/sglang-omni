# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fireredtts3.payload_types import (
    FireRedTTS3State,
    store_fireredtts3_state,
)
from sglang_omni.models.fireredtts3.request_builders import (
    PLACEHOLDER_TOKEN_ID,
    apply_latent_result,
    build_sglang_fireredtts3_request,
)
from sglang_omni.proto import OmniRequest, StagePayload

PATCH_SIZE = 4
PROMPT_FRAMES = 12
TEXT_TOKENS = [5, 6, 7]


def _payload(**overrides) -> StagePayload:
    state = FireRedTTS3State(
        text_token_ids=torch.tensor([TEXT_TOKENS], dtype=torch.long),
        prompt_latents=torch.zeros(1, PROMPT_FRAMES, 64),
        speaker_embedding=torch.zeros(1, 512),
        max_gen_steps=32,
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={}, params={}, metadata={}),
        data={},
    )
    return store_fireredtts3_state(payload, state)


def _build(payload: StagePayload, *, max_sequence_length: int = 4096):
    return build_sglang_fireredtts3_request(
        payload,
        patch_size=PATCH_SIZE,
        max_sequence_length=max_sequence_length,
    )


def test_prefill_ids_reserve_one_row_per_spliced_embedding() -> None:
    data = _build(_payload())

    prompt_patches = PROMPT_FRAMES // PATCH_SIZE
    assert data.prompt_patch_count == prompt_patches
    assert data.text_token_count == len(TEXT_TOKENS)
    ids = data.input_ids[0].tolist()
    # speaker row + text tokens + one row per reference latent patch
    assert (
        ids
        == [PLACEHOLDER_TOKEN_ID, *TEXT_TOKENS]
        + [PLACEHOLDER_TOKEN_ID] * prompt_patches
    )
    assert data.req.origin_input_ids == ids


def test_request_declares_projected_embeddings_and_no_eos() -> None:
    data = _build(_payload())

    assert data.input_embeds_are_projected is True
    assert data.req._input_embeds_are_projected is True
    assert data.req.eos_token_ids == set()
    assert data.max_new_tokens == 32
    assert data.req.sampling_params.max_new_tokens == 32


def test_request_preserves_the_reference_tensors_exactly() -> None:
    latents = torch.randn(1, PROMPT_FRAMES, 64)
    speaker = torch.randn(1, 512)
    data = _build(_payload(prompt_latents=latents, speaker_embedding=speaker))

    # The wire codec transports float32 CPU tensors; the builder must not cast,
    # truncate, or reshape them, because the flow head owns device placement.
    assert torch.equal(data.state.prompt_latents, latents)
    assert torch.equal(data.state.speaker_embedding, speaker)
    assert data.state.prompt_latents.dtype == torch.float32
    assert data.state.speaker_embedding.shape == (1, 512)


def test_request_rejects_unaligned_reference_latents() -> None:
    with pytest.raises(ValueError, match="multiple of 4 frames"):
        _build(_payload(prompt_latents=torch.zeros(1, 10, 64)))


def test_request_requires_preprocessing_and_reference_encoding() -> None:
    with pytest.raises(RuntimeError, match="did not emit text tokens"):
        _build(_payload(text_token_ids=None))
    with pytest.raises(RuntimeError, match="reference encoding did not run"):
        _build(_payload(prompt_latents=None))


def test_request_rejects_a_budget_beyond_the_context_window() -> None:
    with pytest.raises(ValueError, match="engine context length"):
        _build(_payload(), max_sequence_length=16)


def test_result_concatenates_patches_and_reports_usage() -> None:
    payload = _payload()
    data = _build(payload)
    data.latent_patches = [torch.ones(1, PATCH_SIZE, 64) for _ in range(3)]
    data.finish_reason = "stop"

    result = apply_latent_result(data)
    state = FireRedTTS3State.from_dict(result.data)

    assert state.generated_latents.shape == (1, 3 * PATCH_SIZE, 64)
    assert state.completion_tokens == 3
    assert state.prompt_tokens == int(data.input_ids.numel())
    assert state.finish_reason == "stop"
    assert state.engine_time_s > 0.0
    # The reference latents stay for the vocoder; the text tokens are dropped.
    assert state.prompt_latents is not None
    assert state.text_token_ids is None


def test_result_rejects_an_empty_generation() -> None:
    data = _build(_payload())

    with pytest.raises(RuntimeError, match="generated no latent patches"):
        apply_latent_result(data)
