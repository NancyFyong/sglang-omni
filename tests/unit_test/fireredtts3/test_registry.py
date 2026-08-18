# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def test_pipeline_config_is_registered_for_the_published_architecture() -> None:
    from sglang_omni.models.fireredtts3.config import FireRedTTS3PipelineConfig
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FireRedTTS3BaseCore")
        is FireRedTTS3PipelineConfig
    )


def test_sglang_model_is_registered_under_the_arch_override() -> None:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
    from sglang_omni.models.fireredtts3.hf_config import FIREREDTTS3_MODEL_ARCH_OVERRIDE
    from sglang_omni.models.fireredtts3.sglang_model import FireRedTTS3SGLangModel

    SGLModelRunner._register_omni_model(SGLModelRunner)

    assert (
        ModelRegistry.models[FIREREDTTS3_MODEL_ARCH_OVERRIDE] is FireRedTTS3SGLangModel
    )


def test_pipeline_declares_a_non_streaming_reference_cloning_pipeline() -> None:
    from sglang_omni.models.fireredtts3 import CAPABILITIES
    from sglang_omni.models.fireredtts3.config import FireRedTTS3PipelineConfig

    config = FireRedTTS3PipelineConfig(model_path="stub")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "reference_encode",
        "tts_engine",
        "vocoder",
    ]
    assert config.stages[-1].terminal is True
    assert FireRedTTS3PipelineConfig.required_speech_reference_count == 1
    assert FireRedTTS3PipelineConfig.speech_reference_text_required is True
    assert CAPABILITIES.supports_reference_audio is True
    # RedAE decodes with full attention over the reference + generated latents.
    assert CAPABILITIES.supports_streaming_vocoder is False
    assert CAPABILITIES.supports_cuda_graph is False
