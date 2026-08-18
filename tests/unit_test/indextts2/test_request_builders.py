# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.indextts2.hf_config import (
    INDEXTTS2_ARCHITECTURE,
    INDEXTTS2_MODEL_ARCH_OVERRIDE,
    IndexTTS2Config,
)
from sglang_omni.models.indextts2.payload_types import (
    IndexTTS2State,
    store_indextts2_state,
)
from sglang_omni.models.indextts2.request_builders import (
    PLACEHOLDER_TOKEN_ID,
    apply_mel_code_result,
    build_sglang_indextts2_request,
)
from sglang_omni.proto import OmniRequest, StagePayload

TEXT_TOKENS = [11, 12, 13, 14]


def _config() -> IndexTTS2Config:
    return IndexTTS2Config()


def _payload(**overrides) -> StagePayload:
    state = IndexTTS2State(
        text_token_ids=torch.tensor([TEXT_TOKENS], dtype=torch.long),
        speaker_embedding=torch.zeros(1, 192),
        emotion_embedding=torch.zeros(1, 1280),
        prompt_condition=torch.zeros(1, 8, 512),
        reference_mel=torch.zeros(1, 80, 8),
        max_mel_tokens=64,
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={}, params={}, metadata={}),
        data={},
    )
    return store_indextts2_state(payload, state)


def _build(payload: StagePayload):
    return build_sglang_indextts2_request(payload, config=_config())


def test_prefix_reserves_condition_text_and_start_rows() -> None:
    data = _build(_payload())

    config = _config()
    # [speaker+emotion, reserved, reserved] + [start] text [stop]
    assert data.prefix_length == config.condition_tokens + len(TEXT_TOKENS) + 2
    ids = data.input_ids[0].tolist()
    assert ids[:-1] == [PLACEHOLDER_TOKEN_ID] * data.prefix_length
    assert ids[-1] == config.start_mel_token
    assert data.req.origin_input_ids == ids


def test_start_and_stop_text_tokens_are_not_double_counted() -> None:
    config = _config()
    padded = torch.tensor(
        [[config.start_text_token, *TEXT_TOKENS, config.stop_text_token]],
        dtype=torch.long,
    )

    data = _build(_payload(text_token_ids=padded))

    assert data.prefix_length == config.condition_tokens + len(TEXT_TOKENS) + 2


def test_sampling_runs_natively_with_the_model_side_penalty() -> None:
    data = _build(_payload(temperature=0.7, top_p=0.9, top_k=25))
    config = _config()

    params = data.req.sampling_params
    assert (params.temperature, params.top_p, params.top_k) == (0.7, 0.9, 25)
    assert params.max_new_tokens == 64
    # SGLang caps repetition_penalty at 2.0, so the model applies the CTRL
    # penalty itself and the sampler must stay neutral.
    assert params.repetition_penalty == 1.0
    assert data.repetition_penalty == 10.0
    assert data.req.eos_token_ids == {config.stop_mel_token}
    assert data.input_embeds_are_projected is True


def test_request_requires_preprocessing_and_reference_encoding() -> None:
    with pytest.raises(RuntimeError, match="did not emit text tokens"):
        _build(_payload(text_token_ids=None))
    with pytest.raises(RuntimeError, match="reference encoding did not run"):
        _build(_payload(speaker_embedding=None))
    with pytest.raises(RuntimeError, match="no s2mel prompt"):
        _build(_payload(prompt_condition=None))


def test_request_rejects_an_out_of_range_budget() -> None:
    with pytest.raises(ValueError, match="max_mel_tokens must be in"):
        _build(_payload(max_mel_tokens=10_000))


def test_result_strips_the_stop_token() -> None:
    config = _config()
    payload = _payload()
    data = _build(payload)
    data.output_ids.extend([5, 6, 7, config.stop_mel_token])
    data.finish_reason = "stop"

    result = apply_mel_code_result(data, stop_mel_token=config.stop_mel_token)
    state = IndexTTS2State.from_dict(result.data)

    assert state.mel_codes.tolist() == [[5, 6, 7]]
    assert state.completion_tokens == 3
    assert state.finish_reason == "stop"
    # The vocoder still needs the reference conditioning.
    assert state.prompt_condition is not None
    assert state.reference_mel is not None
    assert state.text_token_ids is None


def test_result_rejects_an_empty_generation() -> None:
    config = _config()
    data = _build(_payload())
    data.output_ids.append(config.stop_mel_token)

    with pytest.raises(RuntimeError, match="generated no mel codes"):
        apply_mel_code_result(data, stop_mel_token=config.stop_mel_token)


def test_architectures_are_registered() -> None:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
    from sglang_omni.models.indextts2.config import IndexTTS2PipelineConfig
    from sglang_omni.models.indextts2.sglang_model import IndexTTS2SGLangModel
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    SGLModelRunner._register_omni_model(SGLModelRunner)

    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(INDEXTTS2_ARCHITECTURE)
        is IndexTTS2PipelineConfig
    )
    assert ModelRegistry.models[INDEXTTS2_MODEL_ARCH_OVERRIDE] is IndexTTS2SGLangModel


def test_config_matches_the_upstream_gpt_shape() -> None:
    config = _config()

    assert (config.hidden_size, config.num_hidden_layers) == (1280, 24)
    assert config.num_attention_heads == 20
    assert config.vocab_size == 8194
    assert (config.start_mel_token, config.stop_mel_token) == (8192, 8193)
    # n_positions follows UnifiedVoice.post_init_gpt2_config.
    assert config.n_positions == 1815 + 600 + 2
    assert config.max_mel_positions == 1815 + 2 + 1


def test_state_round_trips_across_the_relay() -> None:
    state = IndexTTS2State.from_dict(_payload().data)
    state.mel_codes = torch.tensor([[1, 2, 3]], dtype=torch.long)

    restored = IndexTTS2State.from_dict(state.to_dict())

    assert torch.equal(restored.mel_codes, state.mel_codes)
    assert torch.equal(restored.prompt_condition, state.prompt_condition)
    assert torch.equal(restored.reference_mel, state.reference_mel)
    assert torch.equal(restored.speaker_embedding, state.speaker_embedding)
    assert torch.equal(restored.emotion_embedding, state.emotion_embedding)
