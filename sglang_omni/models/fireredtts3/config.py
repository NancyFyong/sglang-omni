# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for FireRedTTS3-Base."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.fireredtts3"


class FireRedTTS3PipelineConfig(PipelineConfig):
    """preprocess -> reference encode -> SGLang latent AR -> RedAE decode."""

    architecture: ClassVar[str] = "FireRedTTS3BaseCore"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "FireRedTTS3",
        "FireRedTTS3SGLangModel",
    )
    requires_model_capabilities: ClassVar[bool] = True
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_required: ClassVar[bool] = True
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset({"auto_detect"})

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="reference_encode",
        ),
        StageConfig(
            name="reference_encode",
            process="pipeline",
            factory=f"{_PKG}.stages.create_reference_encode_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"device": "cuda", "precision": "bfloat16"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            terminal=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if any(stage.tp_size != 1 for stage in self.stages):
            raise ValueError("FireRedTTS3 currently supports tp_size=1 only")

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    @classmethod
    def process_safe_edges(cls) -> frozenset[tuple[str, str]]:
        # Every stage rebuilds its inputs from the payload state, and the
        # reference-encode / vocoder stages own their own RedAE instance.
        return frozenset(
            {
                ("preprocessing", "reference_encode"),
                ("tts_engine", "vocoder"),
            }
        )

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = FireRedTTS3PipelineConfig
