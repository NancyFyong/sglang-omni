# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.fireredtts3.payload_types import (
    FireRedTTS3State,
    store_fireredtts3_state,
)
from sglang_omni.models.fireredtts3.vocoder import FireRedTTS3Vocoder
from sglang_omni.proto import OmniRequest, StagePayload

DOWNSAMPLE_RATE = 4
REDAE_DIM = 2


class _StubCodec:
    """Decodes each latent frame into DOWNSAMPLE_RATE samples carrying its index."""

    sample_rate = 24000
    downsample_rate = DOWNSAMPLE_RATE

    def __init__(self) -> None:
        self.seen: list[torch.Tensor] = []

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.seen.append(latents)
        frames = latents.shape[1]
        ramp = torch.arange(frames, dtype=torch.float32).repeat_interleave(
            DOWNSAMPLE_RATE
        )
        return ramp.reshape(1, -1)


def _payload(*, prompt_frames: int = 3, generated_frames: int = 5) -> StagePayload:
    state = FireRedTTS3State(
        prompt_latents=torch.zeros(1, prompt_frames, REDAE_DIM),
        generated_latents=torch.ones(1, generated_frames, REDAE_DIM),
        prompt_tokens=7,
        completion_tokens=generated_frames,
        engine_time_s=0.5,
    )
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={}, params={}, metadata={}),
        data={},
    )
    return store_fireredtts3_state(payload, state)


def test_vocoder_decodes_with_reference_context_then_trims_it() -> None:
    codec = _StubCodec()
    payload = FireRedTTS3Vocoder(codec).decode_payload(_payload())

    # RedAE has full attention, so the reference latents stay in front of the
    # generated ones and only the corresponding samples are dropped.
    assert codec.seen[0].shape == (1, 3 + 5, REDAE_DIM)
    waveform = np.frombuffer(payload.data["audio_waveform"], dtype=np.float32)
    assert payload.data["audio_waveform_shape"] == [5 * DOWNSAMPLE_RATE]
    assert waveform[0] == pytest.approx(3.0)
    assert waveform[-1] == pytest.approx(7.0)


def test_vocoder_emits_the_audio_payload_contract() -> None:
    payload = FireRedTTS3Vocoder(_StubCodec()).decode_payload(_payload())

    assert payload.data["sample_rate"] == 24000
    assert payload.data["audio_waveform_dtype"] == "float32"
    assert payload.data["modality"] == "audio"
    assert payload.data["usage"]["completion_tokens"] == 5


def test_vocoder_releases_the_latent_tensors() -> None:
    payload = FireRedTTS3Vocoder(_StubCodec()).decode_payload(_payload())
    state = FireRedTTS3State.from_dict(payload.data)

    assert state.generated_latents is None
    assert state.prompt_latents is None


def test_vocoder_requires_both_latent_blocks() -> None:
    vocoder = FireRedTTS3Vocoder(_StubCodec())

    with pytest.raises(RuntimeError, match="no generated latents"):
        vocoder.decode_payload(_payload(generated_frames=0))
    with pytest.raises(RuntimeError, match="requires the reference latents"):
        vocoder.decode_payload(_payload(prompt_frames=0))


def test_vocoder_rejects_a_codec_sample_rate_change() -> None:
    class _DriftingCodec(_StubCodec):
        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("FireRedTTS3 RedAE changed its output sample rate")

    with pytest.raises(RuntimeError, match="changed its output sample rate"):
        FireRedTTS3Vocoder(_DriftingCodec()).decode_payload(_payload())


def test_codec_pads_references_to_the_patch_grid() -> None:
    from sglang_omni.models.fireredtts3.codec import FireRedTTS3Codec

    codec = object.__new__(FireRedTTS3Codec)
    codec.sample_rate = 100
    codec.samples_per_patch = 40
    stub = SimpleNamespace(calls=[])

    def _load_audio(source, **kwargs):
        stub.calls.append(kwargs)
        return np.ones(90, dtype=np.float32)

    import sglang_omni.models.fireredtts3.codec as codec_module

    original = codec_module.load_audio
    codec_module.load_audio = _load_audio
    try:
        waveform = codec.load_reference_waveform("ref.wav")
    finally:
        codec_module.load_audio = original

    # Left padding matches RedAE.pad_to_multiple_of so prompt_frames * hop
    # equals the number of samples trimmed after decoding.
    assert waveform.shape == (1, 120)
    assert float(waveform[0, 0]) == 0.0
    assert float(waveform[0, -1]) == 1.0
    assert stub.calls[0]["target_sample_rate"] == 100
