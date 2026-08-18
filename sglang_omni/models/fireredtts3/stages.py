# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the framework-native FireRedTTS3 pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.fireredtts3.codec import (
    FireRedTTS3ReferenceEncoder,
    load_fireredtts3_codec,
)
from sglang_omni.models.fireredtts3.components.text_normalize import (
    build_wetext_normalizer,
    clean_text,
    clean_tn_spaces,
    detect_language,
)
from sglang_omni.models.fireredtts3.components.text_tokenizer import (
    MULTI_DIALECT_TAGS,
    MULTI_LANG_TAGS,
    load_text_tokenizer,
)
from sglang_omni.models.fireredtts3.hf_config import CORE_SUBDIR, TOKENIZER_SUBDIR
from sglang_omni.models.fireredtts3.payload_types import (
    FireRedTTS3State,
    store_fireredtts3_state,
)
from sglang_omni.models.fireredtts3.vocoder import FireRedTTS3Vocoder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

#: Languages/dialects accepted in the ``language`` request field.
SUPPORTED_LANGUAGES = tuple(
    tag.strip("<|>") for tag in (*MULTI_LANG_TAGS, *MULTI_DIALECT_TAGS)
)
_WETEXT_LANGUAGES = frozenset({"Chinese", "English", "Cantonese"})
_DEFAULT_MAX_TEXT_CHARS = 300


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
    raise TypeError("FireRedTTS3 reference audio must be a path or data URI")


def preprocess_fireredtts3_payload(
    payload: StagePayload,
    *,
    tokenizer: Any,
    normalizer: Any,
    max_gen_steps: int,
    max_text_chars: int,
) -> StagePayload:
    inputs = _inputs(payload.request.inputs)
    params = _dict(payload.request.params)
    tts_params = _dict(_dict(payload.request.metadata).get("tts_params"))
    stage_params = _dict(params.get("stage_params"))
    engine_params = _dict(stage_params.get("tts_engine"))
    references = inputs.get("references")
    if isinstance(references, list) and len(references) > 1:
        raise ValueError("FireRedTTS3 accepts at most one reference audio")
    reference = (
        references[0]
        if isinstance(references, list)
        and references
        and isinstance(references[0], dict)
        else {}
    )

    text = str(_first_not_none(inputs.get("input"), inputs.get("text"), default=""))
    text = clean_text(text)
    if not text:
        raise ValueError("FireRedTTS3 requires non-empty input text")
    if len(text) > int(max_text_chars):
        raise ValueError(
            f"FireRedTTS3 input text is {len(text)} characters; this pipeline "
            f"synthesizes one segment per request and accepts up to {max_text_chars}"
        )

    reference_audio = _first_not_none(
        _reference_source(inputs.get("prompt_audio")),
        _reference_source(inputs.get("reference_audio")),
        _reference_source(tts_params.get("ref_audio")),
        _reference_source(reference),
    )
    if not reference_audio:
        raise ValueError("FireRedTTS3 requires a reference audio for voice cloning")
    prompt_text = str(
        _first_not_none(
            inputs.get("prompt_text"),
            inputs.get("reference_text"),
            tts_params.get("ref_text"),
            reference.get("text"),
            default="",
        )
    )

    language_value = _first_not_none(
        inputs.get("language"), tts_params.get("language"), params.get("language")
    )
    language = str(language_value).strip() if language_value else ""
    if not language or language.lower() in {"auto", "auto_detect"}:
        language = detect_language(text)
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"unsupported FireRedTTS3 language {language!r}; "
            f"expected one of {sorted(SUPPORTED_LANGUAGES)}"
        )

    if normalizer is not None and language in _WETEXT_LANGUAGES:
        try:
            text = clean_tn_spaces(normalizer(text))
        except Exception as exc:  # upstream also falls back to the raw text
            logger.warning("FireRedTTS3 text normalization failed: %s", exc)

    prompt_text = clean_text(prompt_text) if prompt_text else ""
    icl_text = f"<|{language}|><|sot|>{prompt_text}{text}<|eot|>"
    token_ids = tokenizer(
        icl_text, truncation=False, padding=False, add_special_tokens=False
    )["input_ids"]

    state = FireRedTTS3State(
        text_token_ids=torch.tensor([token_ids], dtype=torch.long),
        language=language,
        reference_audio=reference_audio,
        n_timesteps=int(
            _first_not_none(
                inputs.get("n_timesteps"),
                tts_params.get("n_timesteps"),
                engine_params.get("n_timesteps"),
                default=10,
            )
        ),
        inference_cfg=float(
            _first_not_none(
                inputs.get("inference_cfg"),
                tts_params.get("inference_cfg"),
                engine_params.get("inference_cfg"),
                default=2.0,
            )
        ),
        stop_threshold=float(
            _first_not_none(
                inputs.get("stop_threshold"),
                tts_params.get("stop_threshold"),
                engine_params.get("stop_threshold"),
                default=0.5,
            )
        ),
        min_gen_steps=int(
            _first_not_none(
                inputs.get("min_gen_steps"),
                tts_params.get("min_gen_steps"),
                engine_params.get("min_gen_steps"),
                default=6,
            )
        ),
        max_gen_steps=int(
            _first_not_none(
                inputs.get("max_gen_steps"),
                tts_params.get("max_gen_steps"),
                params.get("max_new_tokens"),
                default=max_gen_steps,
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
    )
    if state.n_timesteps <= 0:
        raise ValueError("FireRedTTS3 n_timesteps must be positive")
    if not 0.0 < state.stop_threshold <= 1.0:
        raise ValueError("FireRedTTS3 stop_threshold must be in (0, 1]")
    if state.min_gen_steps < 0:
        raise ValueError("FireRedTTS3 min_gen_steps must not be negative")
    if not 0 < state.max_gen_steps <= int(max_gen_steps):
        raise ValueError(
            f"FireRedTTS3 max_gen_steps must be in [1, {max_gen_steps}], "
            f"got {state.max_gen_steps}"
        )
    return store_fireredtts3_state(payload, state)


def create_preprocessing_executor(
    model_path: str,
    *,
    max_gen_steps: int = 400,
    max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS,
    enable_text_normalization: bool = True,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    root = Path(resolve_checkpoint(model_path))
    tokenizer = load_text_tokenizer(str(root / TOKENIZER_SUBDIR))
    normalizer = build_wetext_normalizer() if enable_text_normalization else None
    if enable_text_normalization and normalizer is None:
        logger.warning(
            "FireRedTTS3 text normalization is unavailable (wetext not installed); "
            "sending written-form numbers or units may degrade quality"
        )

    def _preprocess(payload: StagePayload) -> StagePayload:
        return preprocess_fireredtts3_payload(
            payload,
            tokenizer=tokenizer,
            normalizer=normalizer,
            max_gen_steps=max_gen_steps,
            max_text_chars=max_text_chars,
        )

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str | None = "cuda",
    gpu_id: int | None = None,
    patch_size: int = 4,
    max_concurrency: int = 4,
) -> SimpleScheduler:
    codec = load_fireredtts3_codec(
        model_path, device=_device(device, gpu_id), patch_size=patch_size
    )
    encoder = FireRedTTS3ReferenceEncoder(codec, model_id=str(model_path))
    return SimpleScheduler(
        encoder.encode_payload,
        max_concurrency=max_concurrency,
        shutdown_callback=encoder.close,
    )


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
    from sglang_omni.models.fireredtts3.engine_builder import FireRedTTS3EngineBuilder

    return FireRedTTS3EngineBuilder(
        max_running_requests=max_running_requests,
        mem_fraction_static=mem_fraction_static,
    ).build(
        model_path,
        device=device or "cuda",
        gpu_id=gpu_id,
        dtype=precision,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = "cuda",
    gpu_id: int | None = None,
    patch_size: int = 4,
    max_concurrency: int = 2,
    **_: Any,
) -> SimpleScheduler:
    codec = load_fireredtts3_codec(
        model_path, device=_device(device, gpu_id), patch_size=patch_size
    )
    vocoder = FireRedTTS3Vocoder(codec)
    return SimpleScheduler(vocoder.decode_payload, max_concurrency=max_concurrency)


def core_checkpoint_dir(model_path: str) -> Path:
    """Location of the LLM-DiT core inside a FireRedTTS3 checkpoint."""
    return Path(resolve_checkpoint(model_path)) / CORE_SUBDIR


__all__ = [
    "SUPPORTED_LANGUAGES",
    "core_checkpoint_dir",
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
    "preprocess_fireredtts3_payload",
]
