# SPDX-License-Identifier: Apache-2.0
"""IndexTTS-2.5 SGLang engine builder."""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from sglang_omni.models.indextts2.hf_config import (
    GPT_CHECKPOINT,
    INDEXTTS2_MODEL_ARCH_OVERRIDE,
    IndexTTS2Config,
    load_indextts2_config,
    register_indextts2_hf_config,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)


def _build_config_shim(root: Path, config: IndexTTS2Config) -> str:
    """Materialize an HF-loadable directory for the GPT backbone.

    IndexTTS-2.5 ships ``config.yaml`` plus a bare ``gpt.pth``; the shim adds a
    ``config.json`` and links the weights instead of copying them.
    """
    shim = tempfile.mkdtemp(prefix="indextts2_sglang_")
    atexit.register(shutil.rmtree, shim, ignore_errors=True)
    Path(shim, "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    source = root / GPT_CHECKPOINT
    if not source.is_file():
        raise FileNotFoundError(f"IndexTTS-2.5 GPT weights not found: {source}")
    destination = os.path.join(shim, "pytorch_model.bin")
    try:
        os.symlink(source, destination)
    except (OSError, NotImplementedError):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    return shim


class IndexTTS2EngineBuilder(TtsEngineBuilder):
    model_name = "IndexTTS-2.5"
    context_length = 2417

    def __init__(
        self,
        *,
        max_running_requests: int = 8,
        mem_fraction_static: float = 0.35,
    ) -> None:
        self.model_arch_override = INDEXTTS2_MODEL_ARCH_OVERRIDE
        self.max_running_requests = int(max_running_requests)
        self.mem_fraction_static = float(mem_fraction_static)
        if self.max_running_requests <= 0:
            raise ValueError("IndexTTS-2.5 max_running_requests must be positive")
        self.hf_config: IndexTTS2Config | None = None
        self._model_runner: Any | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        root = Path(resolve_checkpoint(model_path))
        self.hf_config = load_indextts2_config(str(root))
        self.context_length = int(self.hf_config.n_positions)
        return _build_config_shim(root, self.hf_config)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        register_indextts2_hf_config()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            # Decode consumes a spliced mel embedding, so the decode graph
            # would replay a stale embedding buffer.
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": self.mem_fraction_static,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("IndexTTS-2.5 support does not implement TP")
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("IndexTTS-2.5 max_running_requests must be positive")
        self.max_running_requests = requested
        # Prefill embeddings are per-request (speaker, emotion, text), so a
        # shared radix prefix would alias two different voices.
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        if not bool(overrides.get("disable_cuda_graph", True)):
            raise ValueError(
                "IndexTTS-2.5 decode consumes spliced mel embeddings; the SGLang "
                "decode CUDA graph is not supported yet"
            )

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id
        model_worker.model_runner.model.eval()
        logger.info(
            "IndexTTS-2.5 mel-code engine: eager GPT2 decode, native SGLang "
            "sampling (max_running_requests=%d)",
            int(server_args.max_running_requests),
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.indextts2.model_runner import IndexTTS2ModelRunner

        self._model_runner = IndexTTS2ModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.indextts2.request_builders import (
            apply_mel_code_result,
            build_sglang_indextts2_request,
        )

        config = model.config
        stop_mel_token = int(config.stop_mel_token)

        def _build_request(payload: Any) -> Any:
            return build_sglang_indextts2_request(payload, config=config)

        def _apply_result(data: Any) -> Any:
            return apply_mel_code_result(data, stop_mel_token=stop_mel_token)

        return _build_request, _apply_result

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {"enable_async_decode": False}


__all__ = ["IndexTTS2EngineBuilder"]
