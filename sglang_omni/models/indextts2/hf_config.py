# SPDX-License-Identifier: Apache-2.0
"""HuggingFace config adapter for IndexTTS-2.5 checkpoints.

IndexTTS-2.5 ships ``config.yaml`` plus loose ``*.pth`` files and no HF
``config.json``, so SGLang cannot load it directly. This module turns the
``gpt:`` section of the YAML into a flat GPT2-style config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import GPT2Config

#: Architecture published by the sglang-omni pipeline config.
INDEXTTS2_ARCHITECTURE = "IndexTTS2ForConditionalGeneration"
#: Class name registered in the SGLang model registry.
INDEXTTS2_MODEL_ARCH_OVERRIDE = "IndexTTS2SGLangModel"

CONFIG_FILENAME = "config.yaml"
GPT_CHECKPOINT = "gpt.pth"
AUX_CACHE_DIR = "hf_cache"
W2V_BERT_DIR = "w2v-bert-2.0"
CAMPPLUS_FILE = "campplus_cn_common.bin"
SEMANTIC_CODEC_FILE = "semantic_codec_model.safetensors"
BIGVGAN_DIR = "bigvgan"

_GPT_FIELDS: dict[str, Any] = {
    "model_dim": 1280,
    "layers": 24,
    "heads": 20,
    "max_mel_tokens": 1815,
    "max_text_tokens": 600,
    "number_text_tokens": 60509,
    "number_mel_codes": 8194,
    "start_mel_token": 8192,
    "stop_mel_token": 8193,
    "start_text_token": 0,
    "stop_text_token": 1,
    "mel_length_compression": 1024,
    "condition_type": "conformer_perceiver",
    "max_conditioning_inputs": 1,
}


class IndexTTS2Config(GPT2Config):
    """GPT2 backbone config plus the IndexTTS-2.5 conditioning fields.

    Subclassing GPT2Config lets SGLang size its KV cache and lets the SGLang
    GPT2 blocks be built from the same object.
    """

    model_type = "indextts2"

    def __init__(self, **kwargs: Any) -> None:
        gpt = {name: kwargs.pop(name, default) for name, default in _GPT_FIELDS.items()}
        self.gpt_fields = gpt
        for name, value in gpt.items():
            setattr(self, name, value)
        # UnifiedVoice.post_init_gpt2_config builds its GPT2Config this way; the
        # published checkpoint has no config.json to read it from.
        sequence_length = int(gpt["max_mel_tokens"]) + int(gpt["max_text_tokens"]) + 2
        kwargs.setdefault("vocab_size", int(gpt["number_mel_codes"]))
        kwargs.setdefault("n_positions", sequence_length)
        kwargs.setdefault("n_ctx", sequence_length)
        kwargs.setdefault("n_embd", int(gpt["model_dim"]))
        kwargs.setdefault("n_layer", int(gpt["layers"]))
        kwargs.setdefault("n_head", int(gpt["heads"]))
        kwargs.setdefault("gradient_checkpointing", False)
        kwargs.setdefault("use_cache", True)
        kwargs.setdefault("architectures", [INDEXTTS2_MODEL_ARCH_OVERRIDE])
        super().__init__(**kwargs)

    # ---- derived prefix geometry -------------------------------------- #
    @property
    def condition_tokens(self) -> int:
        """Speaker+emotion row followed by the two reserved duration rows."""
        return 3

    @property
    def max_prefix_tokens(self) -> int:
        return self.condition_tokens + int(self.max_text_tokens) + 2

    @property
    def max_mel_positions(self) -> int:
        """LearnedPositionEmbeddings size for the mel stream."""
        return int(self.max_mel_tokens) + 2 + int(self.max_conditioning_inputs)


def load_indextts2_config(model_dir: str) -> IndexTTS2Config:
    """Build an IndexTTS2Config from ``<model_dir>/config.yaml``."""
    from omegaconf import OmegaConf

    path = Path(model_dir) / CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"IndexTTS-2.5 checkpoint is missing {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "gpt" not in raw or "s2mel" not in raw:
        raise ValueError(
            f"{path} is not an IndexTTS-2.5 config; expected 'gpt' and 's2mel' sections"
        )
    gpt = raw["gpt"]
    fields = {name: gpt[name] for name in _GPT_FIELDS if name in gpt}
    return IndexTTS2Config(**fields)


def is_indextts2_checkpoint(model_dir: str) -> bool:
    """True when the directory looks like an IndexTTS-2.5 release."""
    root = Path(model_dir)
    if not (root / CONFIG_FILENAME).is_file() or not (root / GPT_CHECKPOINT).is_file():
        return False
    try:
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(OmegaConf.load(root / CONFIG_FILENAME))
    except Exception:
        return False
    return isinstance(raw, dict) and {"gpt", "s2mel", "semantic_codec"} <= set(raw)


def register_indextts2_hf_config() -> None:
    """Register the local config so AutoConfig can load the shim directory."""
    from transformers import AutoConfig

    AutoConfig.register(IndexTTS2Config.model_type, IndexTTS2Config, exist_ok=True)


__all__ = [
    "AUX_CACHE_DIR",
    "BIGVGAN_DIR",
    "CAMPPLUS_FILE",
    "CONFIG_FILENAME",
    "GPT_CHECKPOINT",
    "INDEXTTS2_ARCHITECTURE",
    "INDEXTTS2_MODEL_ARCH_OVERRIDE",
    "IndexTTS2Config",
    "SEMANTIC_CODEC_FILE",
    "W2V_BERT_DIR",
    "is_indextts2_checkpoint",
    "load_indextts2_config",
    "register_indextts2_hf_config",
]
