# SPDX-License-Identifier: Apache-2.0
"""IndexTTS-2.5 model operators outside the SGLang backbone.

The upstream ``indextts`` package owns every operator here; this module only
loads them once per process and exposes the two stage-level entry points:

* :meth:`IndexTTS2Modules.encode_reference` for the reference-encode stage
* :meth:`IndexTTS2Modules.decode_mel_codes` for the vocoder stage
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.models.indextts2.hf_config import (
    AUX_CACHE_DIR,
    BIGVGAN_DIR,
    CAMPPLUS_FILE,
    CONFIG_FILENAME,
    GPT_CHECKPOINT,
    W2V_BERT_DIR,
)
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

#: Upstream truncates reference clips to 15 seconds before encoding.
MAX_REFERENCE_SECONDS = 15.0
#: w2v-BERT hidden layer upstream reads its semantic features from.
SEMANTIC_LAYER = 17
#: Semantic-code frames per s2mel frame; upstream hardcodes this ratio.
S2MEL_LENGTH_RATIO = 1.72
#: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMOTION_BIAS = (0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625)
EMOTION_SUM_LIMIT = 0.8


def normalize_emotion_vector(
    vector: list[float], *, apply_bias: bool = True
) -> list[float]:
    """De-emphasize unstable emotions and cap the total, as upstream does."""
    if len(vector) != len(EMOTION_BIAS):
        raise ValueError(
            f"IndexTTS-2.5 emotion vector needs {len(EMOTION_BIAS)} values "
            f"(happy, angry, sad, afraid, disgusted, melancholic, surprised, calm), "
            f"got {len(vector)}"
        )
    if any(value < 0.0 or value > 1.0 for value in vector):
        raise ValueError("IndexTTS-2.5 emotion vector values must be in [0, 1]")
    values = list(vector)
    if apply_bias:
        values = [value * bias for value, bias in zip(values, EMOTION_BIAS)]
    total = sum(values)
    if total > EMOTION_SUM_LIMIT:
        values = [value * (EMOTION_SUM_LIMIT / total) for value in values]
    return values


class _EmotionConditioner(torch.nn.Module):
    """The ``emo_*`` half of ``UnifiedVoice``, loaded straight from gpt.pth.

    Keeping it out of the SGLang model lets the reference-encode stage turn the
    reference clips into one 1280-dim vector, so the AR stage receives a tiny
    payload instead of two w2v-BERT feature maps.
    """

    def __init__(self, gpt_config: dict[str, Any], model_dim: int) -> None:
        super().__init__()
        from indextts.gpt.conformer_encoder import ConformerEncoder
        from indextts.gpt.perceiver import PerceiverResampler

        module = gpt_config["emo_condition_module"]
        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=module["output_size"],
            linear_units=module["linear_units"],
            attention_heads=module["attention_heads"],
            num_blocks=module["num_blocks"],
            input_layer=module["input_layer"],
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=module["output_size"],
            ff_mult=module["perceiver_mult"],
            heads=module["attention_heads"],
            num_latents=1,
        )
        self.emovec_layer = torch.nn.Linear(1024, model_dim)
        self.emo_layer = torch.nn.Linear(model_dim, model_dim)
        self.emo_cond_mask_pad = torch.nn.ConstantPad1d((1, 0), True)

    def get_emovec(self, features: torch.Tensor) -> torch.Tensor:
        """``(1, frames, 1024)`` semantic features to a ``(1, model_dim)`` vector."""
        # note: upstream passes ``features.shape[-1]`` (the 1024 feature width,
        # not the frame count) as the conformer length. Reproduced because the
        # weights were trained with that mask.
        lengths = torch.tensor([features.shape[-1]], device=features.device)
        encoded, mask = self.emo_conditioning_encoder(features, lengths)
        conds = self.emo_perceiver_encoder(
            encoded, self.emo_cond_mask_pad(mask.squeeze(1))
        )
        return self.emo_layer(self.emovec_layer(conds.squeeze(1)))

    def merge(
        self,
        speaker_features: torch.Tensor,
        emotion_features: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        base = self.get_emovec(speaker_features)
        if emotion_features is speaker_features:
            return base
        emotion = self.get_emovec(emotion_features)
        return base + float(alpha) * (emotion - base)


class IndexTTS2Modules:
    """Every IndexTTS-2.5 operator that runs outside the SGLang backbone."""

    def __init__(self, checkpoint: str, *, device: str) -> None:
        from indextts.codec.models import EnhancedCodec
        from indextts.s2mel.modules.audio import mel_spectrogram
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        from omegaconf import OmegaConf
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

        root = Path(checkpoint)
        cache = root / AUX_CACHE_DIR
        self.device = torch.device(device)
        self.config = OmegaConf.load(root / CONFIG_FILENAME)
        self.lock = threading.RLock()

        w2v_dir = cache / W2V_BERT_DIR
        _require(w2v_dir, "w2v-bert-2.0")
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            str(w2v_dir), local_files_only=True
        )
        self.semantic_model = (
            Wav2Vec2BertModel.from_pretrained(str(w2v_dir), local_files_only=True)
            .to(self.device)
            .eval()
        )
        stats = torch.load(root / self.config.w2v_stat, map_location="cpu")
        self.semantic_mean = stats["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stats["var"]).to(self.device)

        codec_path = root / "codec.pth"
        _require(codec_path, "codec.pth")
        semantic_codec = EnhancedCodec(
            **self.config.semantic_codec, cfg=self.config.semantic_codec
        )
        semantic_codec.load_checkpoint(str(codec_path))
        self.semantic_codec = semantic_codec.to(self.device).eval()

        s2mel = MyModel(self.config.s2mel)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            str(root / self.config.s2mel_checkpoint),
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device).eval()
        self.s2mel.models["cfm"].estimator.setup_caches(
            max_batch_size=1, max_seq_length=8192
        )

        campplus_path = cache / CAMPPLUS_FILE
        _require(campplus_path, CAMPPLUS_FILE)
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_path, map_location="cpu"))
        self.campplus = campplus.to(self.device).eval()

        bigvgan_dir = cache / BIGVGAN_DIR
        _require(bigvgan_dir, BIGVGAN_DIR)
        vocoder = _load_bigvgan(bigvgan_dir)
        vocoder.remove_weight_norm()
        self.bigvgan = vocoder.to(self.device).eval()

        spectrogram = self.config.s2mel["preprocess_params"]["spect_params"]
        fmax = spectrogram.get("fmax", "None")
        self.sample_rate = int(self.config.s2mel["preprocess_params"]["sr"])
        self._mel_kwargs = {
            "n_fft": spectrogram["n_fft"],
            "win_size": spectrogram["win_length"],
            "hop_size": spectrogram["hop_length"],
            "num_mels": spectrogram["n_mels"],
            "sampling_rate": self.sample_rate,
            "fmin": spectrogram.get("fmin", 0),
            "fmax": None if fmax == "None" else 8000,
            "center": False,
        }
        self._mel_spectrogram = mel_spectrogram

        gpt_config = dict(self.config.gpt)
        self.emotion = _EmotionConditioner(gpt_config, int(gpt_config["model_dim"]))
        _load_emotion_weights(self.emotion, root / GPT_CHECKPOINT)
        self.emotion = self.emotion.to(self.device).eval()

        self.emotion_matrix = torch.split(
            torch.load(root / self.config.emo_matrix, map_location=self.device),
            list(self.config.emo_num),
        )
        self.speaker_matrix = torch.split(
            torch.load(root / self.config.spk_matrix, map_location=self.device),
            list(self.config.emo_num),
        )
        self.emotion_counts = list(self.config.emo_num)
        logger.info(
            "IndexTTS-2.5 model operators ready on %s (output %d Hz)",
            device,
            self.sample_rate,
        )

    # ------------------------------------------------------------------ #
    # reference encoding
    # ------------------------------------------------------------------ #
    def _load_reference(self, source: str, sample_rate: int) -> torch.Tensor:
        waveform = load_audio(
            source,
            source_name="IndexTTS-2.5 reference",
            target_sample_rate=sample_rate,
            mono=True,
        )
        audio = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
        if audio.shape[-1] == 0:
            raise ValueError("IndexTTS-2.5 reference audio is empty")
        limit = int(MAX_REFERENCE_SECONDS * sample_rate)
        return audio[:, :limit]

    @torch.inference_mode()
    def semantic_features(self, audio_16k: torch.Tensor) -> torch.Tensor:
        inputs = self.feature_extractor(
            audio_16k, sampling_rate=16000, return_tensors="pt"
        )
        features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.semantic_model(
            input_features=features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[SEMANTIC_LAYER]
        return (hidden - self.semantic_mean) / self.semantic_std

    @torch.inference_mode()
    def encode_reference(
        self, source: str, *, emotion_source: str | None = None
    ) -> dict[str, torch.Tensor]:
        """Encode the speaker clip (and optional emotion clip) for one request."""
        with self.lock:
            audio_16k = self._load_reference(source, 16000)
            audio_22k = self._load_reference(source, self.sample_rate)
            speaker_features = self.semantic_features(audio_16k)
            reference_mel = self._mel_spectrogram(
                audio_22k.to(self.device).float(), **self._mel_kwargs
            )
            style = self.campplus(_kaldi_fbank(audio_16k).to(self.device).unsqueeze(0))
            prompt_condition = self.s2mel.models["length_regulator"](
                speaker_features,
                ylens=torch.LongTensor([reference_mel.size(2)]).to(self.device),
                n_quantizers=3,
                f0=None,
            )[0]
            emotion_features = speaker_features
            if emotion_source is not None and emotion_source != source:
                emotion_features = self.semantic_features(
                    self._load_reference(emotion_source, 16000)
                )
            return {
                "speaker_embedding": style.detach().cpu().float(),
                "speaker_features": speaker_features.detach().cpu().float(),
                "emotion_features": emotion_features.detach().cpu().float(),
                "prompt_condition": prompt_condition.detach().cpu().float(),
                "reference_mel": reference_mel.detach().cpu().float(),
            }

    @torch.inference_mode()
    def emotion_embedding(
        self,
        *,
        speaker_features: torch.Tensor,
        emotion_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
        alpha: float,
        emotion_vector: list[float] | None,
        use_random: bool,
    ) -> torch.Tensor:
        """The 1280-dim emotion vector spliced into the AR prefix."""
        import random

        with self.lock:
            speaker = speaker_features.to(self.device)
            emotion = (
                speaker
                if emotion_features is None
                else emotion_features.to(self.device)
            )
            vector = self.emotion.merge(speaker, emotion, alpha)
            if emotion_vector is None:
                return vector.detach().cpu().float()
            weights = torch.tensor(emotion_vector, device=self.device)
            style = speaker_embedding.to(self.device)
            if use_random:
                indices = [
                    random.randint(0, count - 1) for count in self.emotion_counts
                ]
            else:
                indices = [
                    int(_most_similar(style, matrix)) for matrix in self.speaker_matrix
                ]
            rows = torch.cat(
                [
                    matrix[index].unsqueeze(0)
                    for index, matrix in zip(indices, self.emotion_matrix)
                ],
                dim=0,
            )
            blended = torch.sum(weights.unsqueeze(1) * rows, dim=0).unsqueeze(0)
            merged = blended + (1 - torch.sum(weights)) * vector
            return merged.detach().cpu().float()

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def decode_mel_codes(
        self,
        *,
        mel_codes: torch.Tensor,
        prompt_condition: torch.Tensor,
        reference_mel: torch.Tensor,
        speaker_embedding: torch.Tensor,
        duration_factor: float,
        diffusion_steps: int,
        inference_cfg_rate: float,
    ) -> torch.Tensor:
        """Mel codes to a waveform, dropping the reference prefix."""
        with self.lock:
            codes = mel_codes.to(self.device)
            prompt = prompt_condition.to(self.device)
            reference = reference_mel.to(self.device)
            style = speaker_embedding.to(self.device)
            semantic = self.semantic_codec.decode(codes)
            target_lengths = torch.LongTensor(
                [int(semantic.shape[1] * S2MEL_LENGTH_RATIO * float(duration_factor))]
            ).to(self.device)
            condition = self.s2mel.models["length_regulator"](
                semantic, ylens=target_lengths, n_quantizers=3, f0=None
            )[0]
            merged = torch.cat([prompt, condition], dim=1)
            target = self.s2mel.models["cfm"].inference(
                merged,
                torch.LongTensor([merged.size(1)]).to(self.device),
                reference,
                style,
                None,
                int(diffusion_steps),
                inference_cfg_rate=float(inference_cfg_rate),
            )
            target = target[:, :, reference.size(-1) :]
            waveform = self.bigvgan(target.float()).squeeze().reshape(1, -1)
            return waveform.detach().cpu()


def _load_bigvgan(directory: Path) -> torch.nn.Module:
    """Build BigVGAN from a local snapshot.

    ``BigVGAN.from_pretrained`` goes through the huggingface_hub mixin, whose
    ``_from_pretrained`` signature upstream still expects the removed
    ``proxies``/``resume_download`` arguments. Loading the two local files
    directly avoids that coupling.
    """
    from indextts.s2mel.modules.bigvgan.bigvgan import BigVGAN, load_hparams_from_json

    hparams = load_hparams_from_json(str(directory / "config.json"))
    model = BigVGAN(hparams, use_cuda_kernel=False)
    checkpoint = torch.load(
        directory / "bigvgan_generator.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["generator"])
    return model


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"IndexTTS-2.5 auxiliary model {label} not found at {path}. Run "
            '`python -c "from indextts.utils.model_download import '
            "ensure_models_available; ensure_models_available('<checkpoint>')\""
        )


def _kaldi_fbank(audio_16k: torch.Tensor) -> torch.Tensor:
    import torchaudio

    feature = torchaudio.compliance.kaldi.fbank(
        audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    return feature - feature.mean(dim=0, keepdim=True)


def _most_similar(query: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    similarity = F.cosine_similarity(query.float(), matrix.float(), dim=1)
    return torch.argmax(similarity)


def _load_emotion_weights(module: torch.nn.Module, gpt_path: Path) -> None:
    state = torch.load(gpt_path, map_location="cpu", weights_only=True)
    prefixes = (
        "emo_conditioning_encoder.",
        "emo_perceiver_encoder.",
        "emovec_layer.",
        "emo_layer.",
    )
    selected = {
        name: tensor for name, tensor in state.items() if name.startswith(prefixes)
    }
    mismatch = module.load_state_dict(selected, strict=False)
    if mismatch.unexpected_keys:
        raise RuntimeError(
            f"Unexpected IndexTTS-2.5 emotion weights: {mismatch.unexpected_keys}"
        )
    missing = [
        name for name in mismatch.missing_keys if "emo_cond_mask_pad" not in name
    ]
    if missing:
        raise RuntimeError(f"IndexTTS-2.5 emotion weights missing for {missing}")


_MODULE_CACHE: dict[tuple[str, str], IndexTTS2Modules] = {}
_MODULE_CACHE_LOCK = threading.Lock()


def load_indextts2_modules(model_path: str, *, device: str) -> IndexTTS2Modules:
    checkpoint = str(Path(resolve_checkpoint(model_path)).resolve())
    key = (checkpoint, str(device))
    with _MODULE_CACHE_LOCK:
        modules = _MODULE_CACHE.get(key)
        if modules is None:
            modules = IndexTTS2Modules(checkpoint, device=device)
            _MODULE_CACHE[key] = modules
        return modules


__all__ = [
    "EMOTION_BIAS",
    "IndexTTS2Modules",
    "load_indextts2_modules",
    "normalize_emotion_vector",
]
