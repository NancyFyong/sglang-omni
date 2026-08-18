# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the framework-native IndexTTS-2.5 pipeline."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.indextts2.modules import (
    IndexTTS2Modules,
    load_indextts2_modules,
    normalize_emotion_vector,
)
from sglang_omni.models.indextts2.payload_types import (
    IndexTTS2State,
    load_indextts2_state,
    store_indextts2_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import (
    audio_data_uri_from_reference,
    audio_waveform_payload,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

#: Languages the 2.5 checkpoint was trained on.
SUPPORTED_LANGUAGES = ("zh", "en", "ja", "es", "ar", "zhen")
_NEMO_LANGUAGES = frozenset({"ja", "es"})
_TEXT_NORMALIZED_LANGUAGES = frozenset({"zh", "zhen", "en"})
_LOWERCASE_LANGUAGES = frozenset({"ja", "zh", "zhen", "en"})
_TAG_PATTERN = re.compile(r"<\|([^|]+)\|>")
#: ``<word|reading>`` pronunciation annotations, e.g. ``<行|XING2>``.
_PRONUNCIATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
_KANA_PATTERN = re.compile(r"^[\u3040-\u309f]+$|^[\u30a0-\u30ff]+$")


def _apply_pronunciation_annotations(text: str) -> str:
    """Rewrite ``<word|reading>`` into the checkpoint's special-token form.

    Mirrors ``indextts.infer_v2_5.apply_pronunciation_annotations``; it is
    reimplemented here because importing that module pulls in upstream's
    vendored transformers generation stack, which does not load under the
    transformers version sglang-omni pins.
    """

    def _replace(match: re.Match[str]) -> str:
        word, reading = match.group(1), match.group(2).upper()
        if _KANA_PATTERN.match(reading):
            return f" {reading} "
        token = (
            "SPECIAL_TOKEN_2"
            if re.search(r"[\u4e00-\u9fff]", word)
            else "SPECIAL_TOKEN_1"
        )
        return f"<|{token}|>{reading}<|{token}|>"

    return _PRONUNCIATION_PATTERN.sub(_replace, text)


def _device(device: str | None, gpu_id: int | None) -> str:
    if device is not None and device != "cuda":
        return device
    return f"cuda:{int(gpu_id or 0)}"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _inputs(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"text": value}
    return _dict(value)


def _first_not_none(*values: Any, default: Any = None) -> Any:
    return next((value for value in values if value is not None), default)


def _reference_source(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        source = (
            value.get("audio_path")
            or value.get("path")
            or audio_data_uri_from_reference(value)
        )
        return str(source) if source else None
    raise TypeError("IndexTTS-2.5 reference audio must be a path or data URI")


def _detect_language(text: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    return "en"


class _TextFrontend:
    """Upstream text normalization, G2P, and tiktoken encoding."""

    def __init__(self, model_dir: str) -> None:
        from indextts.utils.front import TextNormalizer
        from indextts.utils.ja_g2p import JapaneseG2PProcessor
        from indextts.utils.tokenizer import get_tokenizer, lang_to_token

        self.tokenizer = get_tokenizer(multilingual=True, model_dir=model_dir)
        self.lang_to_token = lang_to_token
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        glossary = Path(model_dir) / "glossary.yaml"
        if glossary.is_file():
            self.normalizer.load_glossary_from_yaml(str(glossary))
        self.japanese = JapaneseG2PProcessor(g2p_ratio=0)

    def encode(self, text: str, language: str, *, normalize: bool) -> list[int]:
        from indextts.utils.nemo_tn import normalize_text as nemo_normalize

        cleaned = self.normalizer.clean_pattern.sub(
            lambda match: self.normalizer.char_rep_map[match.group()], text
        )
        if normalize:
            if language in _TEXT_NORMALIZED_LANGUAGES:
                cleaned = self.normalizer.normalize(cleaned)
            elif language in _NEMO_LANGUAGES:
                cleaned = nemo_normalize(cleaned, language)
        if language in _LOWERCASE_LANGUAGES:
            cleaned = cleaned.lower()
        elif language == "es":
            cleaned = cleaned.upper()
        cleaned = _apply_pronunciation_annotations(cleaned)
        if language == "ja":
            cleaned = self.japanese.process_ja_text(cleaned)
        cleaned = _TAG_PATTERN.sub(lambda m: f"<|{m.group(1).upper()}|>", cleaned)
        prefix = f"<|{language}|> "
        return list(self.tokenizer.encode(prefix + cleaned, allowed_special="all"))


def preprocess_indextts2_payload(
    payload: StagePayload,
    *,
    frontend: _TextFrontend,
    max_mel_tokens: int,
    max_text_tokens: int,
) -> StagePayload:
    inputs = _inputs(payload.request.inputs)
    params = _dict(payload.request.params)
    tts_params = _dict(_dict(payload.request.metadata).get("tts_params"))
    stage_params = _dict(params.get("stage_params"))
    engine_params = _dict(stage_params.get("tts_engine"))
    references = inputs.get("references")
    if isinstance(references, list) and len(references) > 1:
        raise ValueError("IndexTTS-2.5 accepts at most one reference audio")
    reference = (
        references[0]
        if isinstance(references, list)
        and references
        and isinstance(references[0], dict)
        else {}
    )

    text = str(
        _first_not_none(inputs.get("input"), inputs.get("text"), default="")
    ).strip()
    if not text:
        raise ValueError("IndexTTS-2.5 requires non-empty input text")
    reference_audio = _first_not_none(
        _reference_source(inputs.get("prompt_audio")),
        _reference_source(inputs.get("reference_audio")),
        _reference_source(tts_params.get("ref_audio")),
        _reference_source(reference),
    )
    if not reference_audio:
        raise ValueError("IndexTTS-2.5 requires a reference audio for voice cloning")
    emotion_audio = _first_not_none(
        _reference_source(inputs.get("emotion_audio")),
        _reference_source(tts_params.get("emotion_audio")),
    )

    language_value = _first_not_none(
        inputs.get("language"), tts_params.get("language"), params.get("language")
    )
    language = str(language_value).strip().lower() if language_value else ""
    if not language or language in {"auto", "auto_detect"}:
        language = _detect_language(text)
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"unsupported IndexTTS-2.5 language {language!r}; "
            f"expected one of {sorted(SUPPORTED_LANGUAGES)}"
        )

    token_ids = frontend.encode(
        text,
        language,
        normalize=bool(
            _first_not_none(
                inputs.get("text_normalization"),
                tts_params.get("text_normalization"),
                default=True,
            )
        ),
    )
    if len(token_ids) > int(max_text_tokens):
        raise ValueError(
            f"IndexTTS-2.5 synthesizes one segment per request and accepts up to "
            f"{max_text_tokens} text tokens; got {len(token_ids)}. Split long text "
            "client-side."
        )

    emotion_vector = _first_not_none(
        inputs.get("emotion_vector"), tts_params.get("emotion_vector")
    )
    if emotion_vector is not None:
        emotion_vector = normalize_emotion_vector(
            [float(value) for value in emotion_vector],
            apply_bias=bool(
                _first_not_none(
                    inputs.get("emotion_bias"),
                    tts_params.get("emotion_bias"),
                    default=True,
                )
            ),
        )

    duration_factor = float(
        _first_not_none(
            inputs.get("duration_factor"),
            tts_params.get("duration_factor"),
            default=1.0,
        )
    )
    if not 0.5 <= duration_factor <= 2.0:
        raise ValueError("IndexTTS-2.5 duration_factor must be in [0.5, 2.0]")

    state = IndexTTS2State(
        text_token_ids=torch.tensor([token_ids], dtype=torch.long),
        language=language,
        language_id=int(frontend.lang_to_token(language)),
        reference_audio=reference_audio,
        emotion_audio=emotion_audio,
        emotion_vector=emotion_vector,
        emotion_alpha=float(
            _first_not_none(
                inputs.get("emotion_alpha"),
                tts_params.get("emotion_alpha"),
                default=1.0,
            )
        ),
        emotion_random=bool(
            _first_not_none(
                inputs.get("emotion_random"),
                tts_params.get("emotion_random"),
                default=False,
            )
        ),
        temperature=float(
            _first_not_none(
                inputs.get("temperature"),
                tts_params.get("temperature"),
                engine_params.get("temperature"),
                default=0.8,
            )
        ),
        top_p=float(
            _first_not_none(inputs.get("top_p"), tts_params.get("top_p"), default=0.8)
        ),
        top_k=int(
            _first_not_none(inputs.get("top_k"), tts_params.get("top_k"), default=30)
        ),
        repetition_penalty=float(
            _first_not_none(
                inputs.get("repetition_penalty"),
                tts_params.get("repetition_penalty"),
                default=10.0,
            )
        ),
        max_mel_tokens=int(
            _first_not_none(
                inputs.get("max_mel_tokens"),
                tts_params.get("max_mel_tokens"),
                params.get("max_new_tokens"),
                default=max_mel_tokens,
            )
        ),
        seed=(
            None
            if (
                seed := _first_not_none(
                    inputs.get("seed"), tts_params.get("seed"), params.get("seed")
                )
            )
            is None
            else int(seed)
        ),
        duration_factor=duration_factor,
        diffusion_steps=int(
            _first_not_none(
                inputs.get("diffusion_steps"),
                tts_params.get("diffusion_steps"),
                default=25,
            )
        ),
        inference_cfg_rate=float(
            _first_not_none(
                inputs.get("inference_cfg_rate"),
                tts_params.get("inference_cfg_rate"),
                default=0.7,
            )
        ),
    )
    if not 0.0 < state.emotion_alpha <= 1.0:
        raise ValueError("IndexTTS-2.5 emotion_alpha must be in (0, 1]")
    if state.diffusion_steps <= 0:
        raise ValueError("IndexTTS-2.5 diffusion_steps must be positive")
    if not 0 < state.max_mel_tokens <= int(max_mel_tokens):
        raise ValueError(
            f"IndexTTS-2.5 max_mel_tokens must be in [1, {max_mel_tokens}], "
            f"got {state.max_mel_tokens}"
        )
    return store_indextts2_state(payload, state)


def create_preprocessing_executor(
    model_path: str,
    *,
    max_mel_tokens: int = 1500,
    max_text_tokens: int = 400,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    root = str(resolve_checkpoint(model_path))
    frontend = _TextFrontend(root)

    def _preprocess(payload: StagePayload) -> StagePayload:
        return preprocess_indextts2_payload(
            payload,
            frontend=frontend,
            max_mel_tokens=max_mel_tokens,
            max_text_tokens=max_text_tokens,
        )

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str | None = "cuda",
    gpu_id: int | None = None,
    max_concurrency: int = 2,
) -> SimpleScheduler:
    modules = load_indextts2_modules(model_path, device=_device(device, gpu_id))

    def _encode(payload: StagePayload) -> StagePayload:
        state = load_indextts2_state(payload)
        if not state.reference_audio:
            raise ValueError("IndexTTS-2.5 requires one reference audio")
        artifact = modules.encode_reference(
            state.reference_audio, emotion_source=state.emotion_audio
        )
        state.speaker_embedding = artifact["speaker_embedding"]
        state.prompt_condition = artifact["prompt_condition"]
        state.reference_mel = artifact["reference_mel"]
        state.emotion_embedding = modules.emotion_embedding(
            speaker_features=artifact["speaker_features"],
            emotion_features=artifact["emotion_features"],
            speaker_embedding=artifact["speaker_embedding"],
            alpha=state.emotion_alpha,
            emotion_vector=state.emotion_vector,
            use_random=state.emotion_random,
        )
        state.reference_audio = None
        state.emotion_audio = None
        return store_indextts2_state(payload, state)

    return SimpleScheduler(_encode, max_concurrency=max_concurrency)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    precision: str = "bfloat16",
    max_running_requests: int = 8,
    mem_fraction_static: float = 0.35,
    device: str | None = "cuda",
    gpu_id: int | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> OmniScheduler:
    from sglang_omni.models.indextts2.engine_builder import IndexTTS2EngineBuilder

    return IndexTTS2EngineBuilder(
        max_running_requests=max_running_requests,
        mem_fraction_static=mem_fraction_static,
    ).build(
        model_path,
        device=device or "cuda",
        gpu_id=gpu_id,
        dtype=precision,
        server_args_overrides=server_args_overrides,
    )


def decode_indextts2_payload(
    payload: StagePayload, *, modules: IndexTTS2Modules
) -> StagePayload:
    state = load_indextts2_state(payload)
    if state.mel_codes is None or state.mel_codes.numel() == 0:
        raise RuntimeError("IndexTTS-2.5 vocoder received no mel codes")
    if state.prompt_condition is None or state.reference_mel is None:
        raise RuntimeError("IndexTTS-2.5 vocoder requires the reference conditioning")
    waveform = modules.decode_mel_codes(
        mel_codes=state.mel_codes,
        prompt_condition=state.prompt_condition,
        reference_mel=state.reference_mel,
        speaker_embedding=state.speaker_embedding,
        duration_factor=state.duration_factor,
        diffusion_steps=state.diffusion_steps,
        inference_cfg_rate=state.inference_cfg_rate,
    )
    state.mel_codes = None
    state.prompt_condition = None
    state.reference_mel = None
    state.speaker_embedding = None
    state.emotion_embedding = None
    state.sample_rate = modules.sample_rate
    store_indextts2_state(payload, state)
    payload.data.update(
        audio_waveform_payload(
            waveform.reshape(-1),
            sample_rate=modules.sample_rate,
            modality="audio",
            source_hint="IndexTTS-2.5",
        )
    )
    usage = build_usage(state)
    if usage is not None:
        payload.data["usage"] = usage
    return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = "cuda",
    gpu_id: int | None = None,
    max_concurrency: int = 2,
    **_: Any,
) -> SimpleScheduler:
    modules = load_indextts2_modules(model_path, device=_device(device, gpu_id))

    def _decode(payload: StagePayload) -> StagePayload:
        return decode_indextts2_payload(payload, modules=modules)

    return SimpleScheduler(_decode, max_concurrency=max_concurrency)


__all__ = [
    "SUPPORTED_LANGUAGES",
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
    "decode_indextts2_payload",
    "preprocess_indextts2_payload",
]
