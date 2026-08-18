# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.indextts2.modules import EMOTION_BIAS, normalize_emotion_vector
from sglang_omni.models.indextts2.payload_types import (
    IndexTTS2State,
    store_indextts2_state,
)
from sglang_omni.models.indextts2.stages import (
    decode_indextts2_payload,
    preprocess_indextts2_payload,
)
from sglang_omni.proto import OmniRequest, StagePayload

REFERENCE = "/tmp/reference.wav"


class _StubFrontend:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str, bool]] = []

    def encode(self, text: str, language: str, *, normalize: bool) -> list[int]:
        self.seen.append((text, language, normalize))
        return [10, 11, 12]

    @staticmethod
    def lang_to_token(language: str) -> int:
        return {"zh": 0, "en": 1, "ja": 2, "es": 3, "ar": 4, "zhen": 5}[language]


def _payload(*, inputs=None, tts_params=None, params=None) -> StagePayload:
    base = {
        "input": "hello world",
        "references": [{"audio_path": REFERENCE}],
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


def _run(payload: StagePayload, **kwargs):
    frontend = _StubFrontend()
    options = {
        "frontend": frontend,
        "max_mel_tokens": 1500,
        "max_text_tokens": 400,
    }
    options.update(kwargs)
    result = preprocess_indextts2_payload(payload, **options)
    return IndexTTS2State.from_dict(result.data), frontend


def test_preprocessing_keeps_upstream_defaults() -> None:
    state, frontend = _run(_payload())

    assert frontend.seen == [("hello world", "en", True)]
    assert state.language == "en"
    assert state.language_id == 1
    assert state.reference_audio == REFERENCE
    assert (state.temperature, state.top_p, state.top_k) == (0.8, 0.8, 30)
    assert state.repetition_penalty == 10.0
    assert state.max_mel_tokens == 1500
    assert state.duration_factor == 1.0
    assert (state.diffusion_steps, state.inference_cfg_rate) == (25, 0.7)
    assert state.emotion_vector is None
    assert state.emotion_alpha == 1.0


def test_preprocessing_detects_the_language() -> None:
    assert _run(_payload(inputs={"input": "今天天气很好"}))[0].language == "zh"
    assert _run(_payload(inputs={"input": "こんにちは"}))[0].language == "ja"
    assert _run(_payload(inputs={"input": "مرحبا"}))[0].language == "ar"


def test_preprocessing_rejects_an_unsupported_language() -> None:
    with pytest.raises(ValueError, match="unsupported IndexTTS-2.5 language"):
        _run(_payload(tts_params={"language": "de"}))


def test_preprocessing_requires_text_and_reference() -> None:
    with pytest.raises(ValueError, match="non-empty input text"):
        _run(_payload(inputs={"input": "  "}))
    with pytest.raises(ValueError, match="requires a reference audio"):
        _run(_payload(inputs={"references": []}))


def test_preprocessing_validates_the_control_knobs() -> None:
    with pytest.raises(ValueError, match=r"duration_factor must be in \[0.5, 2.0\]"):
        _run(_payload(tts_params={"duration_factor": 3.0}))
    with pytest.raises(ValueError, match=r"emotion_alpha must be in \(0, 1\]"):
        _run(_payload(tts_params={"emotion_alpha": 1.5}))
    with pytest.raises(ValueError, match="max_mel_tokens must be in"):
        _run(_payload(tts_params={"max_mel_tokens": 5000}))


def test_preprocessing_rejects_text_beyond_one_segment() -> None:
    with pytest.raises(ValueError, match="one segment per request"):
        _run(_payload(), max_text_tokens=2)


def test_preprocessing_normalizes_the_emotion_vector() -> None:
    state, _ = _run(_payload(tts_params={"emotion_vector": [0, 0, 0.8, 0, 0, 0, 0, 0]}))

    # sad has bias 1.0 and the total is already at the 0.8 cap.
    assert state.emotion_vector == pytest.approx([0, 0, 0.8, 0, 0, 0, 0, 0])


def test_emotion_vector_bias_and_cap() -> None:
    biased = normalize_emotion_vector([1.0] * 8)

    assert sum(biased) == pytest.approx(0.8)
    # The relative ordering follows the bias table.
    assert biased[2] > biased[7]
    assert len(EMOTION_BIAS) == 8


def test_emotion_vector_shape_and_range_are_validated() -> None:
    with pytest.raises(ValueError, match="needs 8 values"):
        normalize_emotion_vector([0.5, 0.5])
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        normalize_emotion_vector([2.0] + [0.0] * 7)


class _StubModules:
    sample_rate = 22050

    def __init__(self) -> None:
        self.seen: dict | None = None

    def decode_mel_codes(self, **kwargs):
        self.seen = kwargs
        return torch.arange(6, dtype=torch.float32).reshape(1, -1)


def _decodable_payload() -> StagePayload:
    state = IndexTTS2State(
        mel_codes=torch.tensor([[1, 2, 3]], dtype=torch.long),
        prompt_condition=torch.zeros(1, 4, 512),
        reference_mel=torch.zeros(1, 80, 4),
        speaker_embedding=torch.zeros(1, 192),
        emotion_embedding=torch.zeros(1, 1280),
        completion_tokens=3,
        duration_factor=1.2,
        diffusion_steps=10,
        inference_cfg_rate=0.5,
    )
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={}, params={}, metadata={}),
        data={},
    )
    return store_indextts2_state(payload, state)


def test_vocoder_forwards_the_request_knobs_and_emits_audio() -> None:
    modules = _StubModules()

    payload = decode_indextts2_payload(_decodable_payload(), modules=modules)

    assert modules.seen["duration_factor"] == pytest.approx(1.2)
    assert modules.seen["diffusion_steps"] == 10
    assert modules.seen["inference_cfg_rate"] == pytest.approx(0.5)
    assert payload.data["sample_rate"] == 22050
    assert payload.data["audio_waveform_dtype"] == "float32"
    waveform = np.frombuffer(payload.data["audio_waveform"], dtype=np.float32)
    assert waveform.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert payload.data["usage"]["completion_tokens"] == 3


def test_vocoder_releases_the_conditioning_tensors() -> None:
    payload = decode_indextts2_payload(_decodable_payload(), modules=_StubModules())
    state = IndexTTS2State.from_dict(payload.data)

    assert state.mel_codes is None
    assert state.prompt_condition is None
    assert state.reference_mel is None
    assert state.speaker_embedding is None


def test_vocoder_requires_codes_and_conditioning() -> None:
    payload = _decodable_payload()
    state = IndexTTS2State.from_dict(payload.data)
    state.mel_codes = None
    with pytest.raises(RuntimeError, match="received no mel codes"):
        decode_indextts2_payload(
            store_indextts2_state(payload, state), modules=_StubModules()
        )

    payload = _decodable_payload()
    state = IndexTTS2State.from_dict(payload.data)
    state.prompt_condition = None
    with pytest.raises(RuntimeError, match="requires the reference conditioning"):
        decode_indextts2_payload(
            store_indextts2_state(payload, state), modules=_StubModules()
        )
