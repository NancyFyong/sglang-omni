# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 RedAE codec and CAM++ speaker encoder.

Both the reference-encode stage and the vocoder stage load this bundle: the
former needs ``RedAE.encode`` plus the speaker embedding, the latter needs
``RedAE.decode``.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang_omni.models.fireredtts3.components.campp import CamppEmbedding
from sglang_omni.models.fireredtts3.components.redae import RedAE
from sglang_omni.models.fireredtts3.hf_config import CAMPPLUS_RELPATH, REDAE_SUBDIR
from sglang_omni.models.fireredtts3.payload_types import (
    load_fireredtts3_state,
    store_fireredtts3_state,
)
from sglang_omni.preprocessing.cache_key import (
    hash_media_item,
    reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)


class FireRedTTS3Codec:
    """RedAE autoencoder plus the CAM++ speaker encoder."""

    def __init__(self, checkpoint: str, *, device: str, patch_size: int) -> None:
        root = Path(checkpoint)
        redae_dir = root / REDAE_SUBDIR
        campplus_path = root / CAMPPLUS_RELPATH
        for path in (redae_dir, campplus_path):
            if not path.exists():
                raise FileNotFoundError(f"FireRedTTS3 checkpoint is missing {path}")
        self.device = torch.device(device)
        # Upstream keeps RedAE in float32 and only autocasts its encoder, so the
        # decoded waveform stays bit-comparable with the reference pipeline.
        self.redae = RedAE.from_pretrained(str(redae_dir)).to(self.device).eval()
        self.speaker = CamppEmbedding(str(campplus_path)).to(self.device).eval()
        self.patch_size = int(patch_size)
        self.sample_rate = int(self.redae.sample_rate)
        self.downsample_rate = int(self.redae.downsample_rate)
        self.samples_per_patch = self.downsample_rate * self.patch_size
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # reference encoding
    # ------------------------------------------------------------------ #
    def load_reference_waveform(self, source: str) -> torch.Tensor:
        waveform = load_audio(
            source,
            source_name="FireRedTTS3 reference",
            target_sample_rate=self.sample_rate,
            mono=True,
        )
        audio = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
        if audio.shape[-1] == 0:
            raise ValueError("FireRedTTS3 reference audio is empty")
        target = math.ceil(audio.shape[-1] / self.samples_per_patch) * (
            self.samples_per_patch
        )
        # RedAE.pad_to_multiple_of left-pads; keep the same alignment so the
        # prompt frame count matches the samples trimmed after decoding.
        return F.pad(audio, (target - audio.shape[-1], 0))

    @torch.inference_mode()
    def encode_reference(self, source: str) -> dict[str, torch.Tensor]:
        audio = self.load_reference_waveform(source).to(self.device)
        with self.lock:
            latents = self.redae.encode(audio, self.sample_rate).to(torch.float32)
            speaker = self.speaker.forward(audio, self.sample_rate)
        frames = int(latents.shape[1])
        if frames * self.downsample_rate != audio.shape[-1]:
            raise RuntimeError(
                f"FireRedTTS3 reference encode produced {frames} frames for "
                f"{audio.shape[-1]} samples; expected a "
                f"{self.downsample_rate}-sample hop"
            )
        return {
            "prompt_latents": latents.detach().cpu().float(),
            "speaker_embedding": speaker.detach().cpu().float().reshape(1, -1),
        }

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``(1, frames, redae_dim)`` latents into a ``(1, samples)`` waveform."""
        with self.lock:
            audio, sample_rate = self.redae.decode(
                latents.to(device=self.device, dtype=torch.float32)
            )
        if int(sample_rate) != self.sample_rate:
            raise RuntimeError("FireRedTTS3 RedAE changed its output sample rate")
        return audio


_CODEC_CACHE: dict[tuple[str, str, int], FireRedTTS3Codec] = {}
_CODEC_CACHE_LOCK = threading.Lock()


def load_fireredtts3_codec(
    model_path: str, *, device: str, patch_size: int
) -> FireRedTTS3Codec:
    checkpoint = str(Path(resolve_checkpoint(model_path)).resolve())
    key = (checkpoint, str(device), int(patch_size))
    with _CODEC_CACHE_LOCK:
        codec = _CODEC_CACHE.get(key)
        if codec is None:
            codec = FireRedTTS3Codec(
                checkpoint, device=device, patch_size=int(patch_size)
            )
            _CODEC_CACHE[key] = codec
        return codec


class _FireRedReferenceHook(KeyedReferenceEncodeHook[str, dict, dict]):
    model_revision = ""
    encoder_id = "fireredtts3_redae_campplus"
    artifact_kind = "reference_conditioning"

    def __init__(self, codec: FireRedTTS3Codec, *, model_id: str) -> None:
        self.codec = codec
        self.model_id = model_id
        self.encoder_config_hash = (
            f"sr{codec.sample_rate}:hop{codec.downsample_rate}:"
            f"patch{codec.patch_size}"
        )

    def input_key(self, item: str) -> str | None:
        return reference_path_cache_key(item, trust_stat=False) or hash_media_item(item)

    def encode_one(self, item: str) -> dict:
        return self.codec.encode_reference(item)

    @staticmethod
    def store_artifact(artifact: dict) -> dict:
        return {
            name: tensor.detach().cpu().float().clone()
            for name, tensor in artifact.items()
        }

    @staticmethod
    def load_artifact(stored: dict) -> dict:
        return {name: tensor.clone() for name, tensor in stored.items()}


class FireRedTTS3ReferenceEncoder:
    """Reference-encode stage worker with a shared artifact cache."""

    def __init__(self, codec: FireRedTTS3Codec, *, model_id: str) -> None:
        self.codec = codec
        self.service = ReferenceEncodeService(
            _FireRedReferenceHook(codec, model_id=model_id),
            max_items=256,
            max_bytes=256 * 1024 * 1024,
            log_prefix="FireRedTTS3",
        )

    def close(self) -> None:
        self.service.close()

    def encode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_fireredtts3_state(payload)
        source = state.reference_audio
        if not source:
            raise ValueError("FireRedTTS3 requires one reference audio")
        artifact = self.service.get_or_encode(source, desc="FireRedTTS3 reference")
        state.prompt_latents = artifact["prompt_latents"]
        state.speaker_embedding = artifact["speaker_embedding"]
        state.reference_audio = None
        return store_fireredtts3_state(payload, state)


__all__ = [
    "FireRedTTS3Codec",
    "FireRedTTS3ReferenceEncoder",
    "load_fireredtts3_codec",
]
