# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fireredtts3.payload_types import FireRedTTS3State
from sglang_omni.models.fireredtts3.stages import preprocess_fireredtts3_payload
from sglang_omni.proto import OmniRequest, StagePayload

REFERENCE = "/tmp/reference.wav"


class _StubTokenizer:
    """Records the ICL string and returns one id per character."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, text: str, **kwargs):
        assert kwargs["add_special_tokens"] is False
        self.seen.append(text)
        return {"input_ids": list(range(len(text)))}


def _payload(
    *,
    inputs: dict | None = None,
    tts_params: dict | None = None,
    params: dict | None = None,
) -> StagePayload:
    base = {
        "input": "hello world",
        "references": [{"audio_path": REFERENCE, "text": "reference transcript"}],
    }
    base.update(inputs or {})
    return StagePayload(
        request_id="rid",
        request=OmniRequest(
            inputs=base,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def _run(payload: StagePayload, **kwargs) -> tuple[FireRedTTS3State, _StubTokenizer]:
    tokenizer = _StubTokenizer()
    options = {
        "tokenizer": tokenizer,
        "normalizer": None,
        "max_gen_steps": 400,
        "max_text_chars": 300,
    }
    options.update(kwargs)
    result = preprocess_fireredtts3_payload(payload, **options)
    return FireRedTTS3State.from_dict(result.data), tokenizer


def test_preprocessing_builds_the_upstream_icl_prompt() -> None:
    state, tokenizer = _run(_payload())

    assert tokenizer.seen == [
        "<|English|><|sot|>reference transcripthello world<|eot|>"
    ]
    assert state.language == "English"
    assert state.reference_audio == REFERENCE
    assert state.text_token_ids is not None
    assert state.text_token_ids.shape == (1, len(tokenizer.seen[0]))


def test_preprocessing_auto_detects_chinese_without_fasttext() -> None:
    state, tokenizer = _run(_payload(inputs={"input": "今天天气很好"}))

    assert state.language == "Chinese"
    assert tokenizer.seen[0].startswith("<|Chinese|><|sot|>")


def test_preprocessing_honours_an_explicit_dialect_tag() -> None:
    state, _ = _run(_payload(tts_params={"language": "ZH_Sichuan"}))

    assert state.language == "ZH_Sichuan"


def test_preprocessing_rejects_an_unsupported_language() -> None:
    with pytest.raises(ValueError, match="unsupported FireRedTTS3 language"):
        _run(_payload(tts_params={"language": "Klingon"}))


def test_preprocessing_requires_reference_audio() -> None:
    payload = _payload(inputs={"references": []})

    with pytest.raises(ValueError, match="requires a reference audio"):
        _run(payload)


def test_preprocessing_requires_non_empty_text() -> None:
    with pytest.raises(ValueError, match="non-empty input text"):
        _run(_payload(inputs={"input": "   "}))


def test_preprocessing_rejects_text_beyond_one_segment() -> None:
    with pytest.raises(ValueError, match="synthesizes one segment per request"):
        _run(_payload(inputs={"input": "x" * 40}), max_text_chars=32)


def test_preprocessing_keeps_upstream_sampling_defaults() -> None:
    state, _ = _run(_payload())

    assert state.n_timesteps == 10
    assert state.inference_cfg == pytest.approx(2.0)
    assert state.stop_threshold == pytest.approx(0.5)
    assert state.min_gen_steps == 6
    assert state.max_gen_steps == 400
    assert state.seed is None


def test_preprocessing_prefers_request_overrides_over_defaults() -> None:
    state, _ = _run(
        _payload(
            tts_params={
                "n_timesteps": 4,
                "inference_cfg": 1.2,
                "stop_threshold": 0.8,
                "seed": 7,
            },
            params={"stage_params": {"tts_engine": {"min_gen_steps": 2}}},
        )
    )

    assert state.n_timesteps == 4
    assert state.inference_cfg == pytest.approx(1.2)
    assert state.stop_threshold == pytest.approx(0.8)
    assert state.min_gen_steps == 2
    assert state.seed == 7


def test_preprocessing_caps_the_generation_budget() -> None:
    with pytest.raises(ValueError, match=r"max_gen_steps must be in \[1, 400\]"):
        _run(_payload(tts_params={"max_gen_steps": 401}))


def test_preprocessing_accepts_a_data_uri_reference() -> None:
    payload = _payload(
        inputs={
            "references": [
                {"data": "UklGRg==", "media_type": "audio/wav", "text": "hi"}
            ]
        }
    )

    state, _ = _run(payload)

    assert state.reference_audio == "data:audio/wav;base64,UklGRg=="


def test_preprocessing_rejects_multiple_references() -> None:
    payload = _payload(
        inputs={
            "references": [
                {"audio_path": REFERENCE, "text": "a"},
                {"audio_path": REFERENCE, "text": "b"},
            ]
        }
    )

    with pytest.raises(ValueError, match="at most one reference audio"):
        _run(payload)


def test_state_round_trips_across_the_relay() -> None:
    state, _ = _run(_payload(tts_params={"seed": 3}))
    state.prompt_latents = torch.zeros(1, 8, 64)
    state.speaker_embedding = torch.zeros(1, 512)

    restored = FireRedTTS3State.from_dict(state.to_dict())

    assert restored.language == state.language
    assert restored.seed == 3
    assert restored.reference_audio == REFERENCE
    assert torch.equal(restored.prompt_latents, state.prompt_latents)
    assert torch.equal(restored.speaker_embedding, state.speaker_embedding)
    assert torch.equal(restored.text_token_ids, state.text_token_ids)
