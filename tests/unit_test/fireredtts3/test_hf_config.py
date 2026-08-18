# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest


def _write_core(tmp_path, **overrides):
    core = tmp_path / "fireredtts3_base"
    core.mkdir()
    payload = {
        "architectures": ["FireRedTTS3BaseCore"],
        "redae_dim": 64,
        "patch_size": 4,
        "dit_depth": 11,
        "num_history_patches": 2,
        "spk_in_dim": 512,
    }
    payload.update(overrides)
    (core / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return core


def test_config_supplies_backbone_shape_missing_from_the_checkpoint(tmp_path) -> None:
    from sglang_omni.models.fireredtts3.hf_config import load_fireredtts3_config

    config = load_fireredtts3_config(str(_write_core(tmp_path)))

    # The published config.json carries none of these; upstream hardcodes them.
    assert config.model_type == "fireredtts3"
    assert (config.hidden_size, config.num_hidden_layers) == (2048, 28)
    assert (config.num_attention_heads, config.num_key_value_heads) == (16, 8)
    assert config.head_dim == 128
    assert config.vocab_size == 151936
    assert config.tie_word_embeddings is True
    # transformers 5.x folds rope_theta into rope_parameters; the Qwen3 default
    # is 10000, so a dropped override would silently change the backbone.
    assert config.rope_parameters["rope_theta"] == 1000000


def test_config_derives_flow_head_shapes(tmp_path) -> None:
    from sglang_omni.models.fireredtts3.hf_config import load_fireredtts3_config

    config = load_fireredtts3_config(str(_write_core(tmp_path)))

    assert config.history_length == 8
    # DiT consumes the latent patch, the speaker condition, and the LLM cond.
    assert config.dit_in_channels == 64 + 512 + 1024


def test_config_keeps_checkpoint_core_overrides(tmp_path) -> None:
    from sglang_omni.models.fireredtts3.hf_config import load_fireredtts3_config

    config = load_fireredtts3_config(str(_write_core(tmp_path, patch_size=8)))

    assert config.patch_size == 8
    assert config.history_length == 16


def test_config_rejects_a_foreign_checkpoint(tmp_path) -> None:
    from sglang_omni.models.fireredtts3.hf_config import load_fireredtts3_config

    core = _write_core(tmp_path, architectures=["Qwen3ForCausalLM"])

    with pytest.raises(ValueError, match="not a FireRedTTS3 core checkpoint"):
        load_fireredtts3_config(str(core))


def test_registered_config_round_trips_through_autoconfig(tmp_path) -> None:
    from transformers import AutoConfig

    from sglang_omni.models.fireredtts3.hf_config import (
        FireRedTTS3Config,
        load_fireredtts3_config,
        register_fireredtts3_hf_config,
    )

    config = load_fireredtts3_config(str(_write_core(tmp_path)))
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")

    register_fireredtts3_hf_config()
    reloaded = AutoConfig.from_pretrained(shim, local_files_only=True)

    assert isinstance(reloaded, FireRedTTS3Config)
    assert reloaded.patch_size == 4
    assert reloaded.dit_hidden_size == 1024
    assert reloaded.hidden_size == 2048


def test_nested_config_resolution_finds_the_core_directory(tmp_path) -> None:
    from sglang_omni.utils.hf import try_resolve_arch_from_nested_config

    _write_core(tmp_path)

    # The published repository root has no config.json at all.
    assert not (tmp_path / "config.json").exists()
    assert try_resolve_arch_from_nested_config(str(tmp_path)) == "FireRedTTS3BaseCore"


def test_nested_config_resolution_ignores_unknown_subdirectories(tmp_path) -> None:
    from sglang_omni.utils.hf import try_resolve_arch_from_nested_config

    other = tmp_path / "some_encoder"
    other.mkdir()
    (other / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"]}), encoding="utf-8"
    )

    assert try_resolve_arch_from_nested_config(str(tmp_path)) is None
