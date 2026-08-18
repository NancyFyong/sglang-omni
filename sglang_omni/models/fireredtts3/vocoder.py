# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 vocoder stage: RedAE latents to waveform."""

from __future__ import annotations

import torch

from sglang_omni.models.fireredtts3.codec import FireRedTTS3Codec
from sglang_omni.models.fireredtts3.payload_types import (
    load_fireredtts3_state,
    store_fireredtts3_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.utils.audio_payload import audio_waveform_payload


class FireRedTTS3Vocoder:
    """Decode reference + generated latents, then drop the reference prefix.

    RedAE decodes the whole latent sequence with full attention, so the
    reference latents stay in front of the generated ones (as upstream does) and
    the corresponding samples are trimmed afterwards.
    """

    def __init__(self, codec: FireRedTTS3Codec) -> None:
        self.codec = codec

    def decode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_fireredtts3_state(payload)
        generated = state.generated_latents
        prompt = state.prompt_latents
        if generated is None or generated.numel() == 0:
            raise RuntimeError("FireRedTTS3 vocoder received no generated latents")
        if prompt is None or prompt.numel() == 0:
            raise RuntimeError("FireRedTTS3 vocoder requires the reference latents")
        latents = torch.cat(
            [prompt.to(generated.dtype), generated.to(generated.dtype)], dim=1
        )
        waveform = self.codec.decode(latents)
        prompt_samples = int(prompt.shape[1]) * self.codec.downsample_rate
        waveform = waveform[:, prompt_samples:].detach().cpu().reshape(-1)
        state.generated_latents = None
        state.prompt_latents = None
        state.sample_rate = self.codec.sample_rate
        store_fireredtts3_state(payload, state)
        payload.data.update(
            audio_waveform_payload(
                waveform,
                sample_rate=self.codec.sample_rate,
                modality="audio",
                source_hint="FireRedTTS3",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


__all__ = ["FireRedTTS3Vocoder"]
