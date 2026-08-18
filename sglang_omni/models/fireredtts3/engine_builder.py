# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 SGLang engine builder."""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from sglang_omni.models.fireredtts3.hf_config import (
    CORE_SUBDIR,
    FIREREDTTS3_MODEL_ARCH_OVERRIDE,
    FireRedTTS3Config,
    load_fireredtts3_config,
    register_fireredtts3_hf_config,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

_WEIGHT_FILE = "model.safetensors"


def _build_config_shim(core_dir: Path, config: FireRedTTS3Config) -> str:
    """Materialize an HF-loadable directory for the LLM-DiT core.

    ``fireredtts3_base/config.json`` carries neither ``model_type`` nor the
    backbone shape, so SGLang cannot load it directly. The shim adds both and
    links the weights instead of copying 8 GB.
    """
    shim = tempfile.mkdtemp(prefix="fireredtts3_sglang_")
    atexit.register(shutil.rmtree, shim, ignore_errors=True)
    payload = config.to_dict()
    payload["architectures"] = [FIREREDTTS3_MODEL_ARCH_OVERRIDE]
    Path(shim, "config.json").write_text(json.dumps(payload), encoding="utf-8")
    source = core_dir / _WEIGHT_FILE
    if not source.is_file():
        raise FileNotFoundError(f"FireRedTTS3 core weights not found: {source}")
    destination = os.path.join(shim, _WEIGHT_FILE)
    try:
        os.symlink(source, destination)
    except (OSError, NotImplementedError):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    return shim


class FireRedTTS3EngineBuilder(TtsEngineBuilder):
    model_name = "FireRedTTS3"
    context_length = 4096

    def __init__(
        self,
        *,
        max_running_requests: int = 8,
        mem_fraction_static: float = 0.35,
    ) -> None:
        self.model_arch_override = FIREREDTTS3_MODEL_ARCH_OVERRIDE
        self.max_running_requests = int(max_running_requests)
        self.mem_fraction_static = float(mem_fraction_static)
        if self.max_running_requests <= 0:
            raise ValueError("FireRedTTS3 max_running_requests must be positive")
        self.hf_config: FireRedTTS3Config | None = None
        self._model_runner: Any | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        root = Path(resolve_checkpoint(model_path))
        core_dir = root / CORE_SUBDIR
        if not core_dir.is_dir():
            raise FileNotFoundError(
                f"FireRedTTS3 checkpoint is missing {CORE_SUBDIR}/: {root}"
            )
        self.hf_config = load_fireredtts3_config(str(core_dir))
        return _build_config_shim(core_dir, self.hf_config)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        register_fireredtts3_hf_config()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            # The decode step feeds a continuous embedding produced by the DiT
            # flow head, so the SGLang decode graph cannot be captured yet.
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
            raise ValueError("FireRedTTS3 support does not implement TP")
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("FireRedTTS3 max_running_requests must be positive")
        self.max_running_requests = requested
        # Prefill embeddings are spliced per position, so a shared radix prefix
        # would alias two different references onto one KV block.
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        if not bool(overrides.get("disable_cuda_graph", True)):
            raise ValueError(
                "FireRedTTS3 decode consumes flow-head embeddings; the SGLang "
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
        model = model_worker.model_runner.model
        model.eval()
        logger.info(
            "FireRedTTS3 latent engine: eager backbone decode, fp32 DiT flow head "
            "(max_running_requests=%d)",
            int(server_args.max_running_requests),
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.fireredtts3.model_runner import FireRedTTS3ModelRunner

        self._model_runner = FireRedTTS3ModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.fireredtts3.request_builders import (
            apply_latent_result,
            build_sglang_fireredtts3_request,
        )

        patch_size = int(model.head.patch_size)
        context_length = self.context_length

        def _build_request(payload: Any) -> Any:
            return build_sglang_fireredtts3_request(
                payload,
                patch_size=patch_size,
                max_sequence_length=context_length,
            )

        return _build_request, apply_latent_result

    def make_abort_callback(self) -> Any | None:
        assert self._model_runner is not None
        return self._model_runner.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {"enable_async_decode": False}


__all__ = ["FireRedTTS3EngineBuilder"]
