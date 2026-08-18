# SPDX-License-Identifier: Apache-2.0
"""HuggingFace config adapter for FireRedTTS3 checkpoints.

The published checkpoint stores the LLM-DiT core under ``fireredtts3_base/``
with a ``config.json`` that carries neither ``model_type`` nor the backbone
Qwen3 shape (upstream hardcodes the Qwen3-1.7B dict in Python). This module
rebuilds a single flat HF config so SGLang can size its KV cache and build the
backbone from it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import Qwen3Config

#: Architecture published in ``fireredtts3_base/config.json``.
FIREREDTTS3_ARCHITECTURE = "FireRedTTS3BaseCore"
#: Class name registered in the SGLang model registry.
FIREREDTTS3_MODEL_ARCH_OVERRIDE = "FireRedTTS3SGLangModel"
#: Checkpoint subdirectory holding the LLM-DiT core.
CORE_SUBDIR = "fireredtts3_base"
#: Sibling subdirectories consumed by the reference-encode and vocoder stages.
REDAE_SUBDIR = "redae"
TOKENIZER_SUBDIR = "text_tokenizer"
CAMPPLUS_RELPATH = "campp/campplus_voxceleb.bin"

# Mirrors fireredtts3.llm.fireredtts3_base.Qwen3_1_7B_ConfigDict; the checkpoint
# does not ship it, so a drift here silently loads a mis-shaped backbone.
_BACKBONE_DEFAULTS: dict[str, Any] = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 28,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

_CORE_DEFAULTS: dict[str, Any] = {
    "redae_dim": 64,
    "num_history_patches": 2,
    "spk_in_dim": 512,
    "patch_size": 4,
    "patch_encoder_hidden_size": 1024,
    "patch_encoder_mlp_ratio": 4,
    "patch_encoder_depth": 8,
    "patch_encoder_num_heads": 16,
    "dit_mlp_ratio": 3,
    "dit_depth": 11,
    "dit_num_heads": 16,
    "dit_hidden_size": 1024,
}


class FireRedTTS3Config(Qwen3Config):
    """Qwen3 backbone config extended with the FireRedTTS3 core fields.

    Subclassing Qwen3Config keeps ``Qwen3ForCausalLM`` and SGLang's
    ``ModelConfig`` working off the same object instead of a nested sub-config.
    """

    model_type = "fireredtts3"

    def __init__(self, **kwargs: Any) -> None:
        for key, default in _CORE_DEFAULTS.items():
            setattr(self, key, kwargs.pop(key, default))
        for key, default in _BACKBONE_DEFAULTS.items():
            kwargs.setdefault(key, default)
        kwargs.setdefault("architectures", [FIREREDTTS3_MODEL_ARCH_OVERRIDE])
        super().__init__(**kwargs)

    @property
    def history_length(self) -> int:
        return int(self.num_history_patches) * int(self.patch_size)

    @property
    def dit_in_channels(self) -> int:
        return int(self.redae_dim) + int(self.spk_in_dim) + int(self.dit_hidden_size)


def load_fireredtts3_config(core_dir: str) -> FireRedTTS3Config:
    """Build a FireRedTTS3Config from ``<core_dir>/config.json``."""
    raw = json.loads((Path(core_dir) / "config.json").read_text(encoding="utf-8"))
    published = raw.get("architectures") or []
    if FIREREDTTS3_ARCHITECTURE not in published:
        raise ValueError(
            f"{core_dir} is not a FireRedTTS3 core checkpoint; "
            f"architectures={published}"
        )
    fields = {key: raw[key] for key in _CORE_DEFAULTS if key in raw}
    return FireRedTTS3Config(**fields)


def register_fireredtts3_hf_config() -> None:
    """Register the local config so AutoConfig can load the shim directory."""
    from transformers import AutoConfig

    AutoConfig.register(FireRedTTS3Config.model_type, FireRedTTS3Config, exist_ok=True)


__all__ = [
    "CAMPPLUS_RELPATH",
    "CORE_SUBDIR",
    "FIREREDTTS3_ARCHITECTURE",
    "FIREREDTTS3_MODEL_ARCH_OVERRIDE",
    "FireRedTTS3Config",
    "REDAE_SUBDIR",
    "TOKENIZER_SUBDIR",
    "load_fireredtts3_config",
    "register_fireredtts3_hf_config",
]
