# SPDX-License-Identifier: Apache-2.0
"""Omni ModelRunner hooks for FireRedTTS3 continuous latent feedback."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner


class FireRedTTS3ModelRunner(ModelRunner):
    """Use the shared SGLang forward path and own only latent recurrence."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._request_data: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # prefill
    # ------------------------------------------------------------------ #
    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        if not requests:
            return
        head = self.model.head
        device = forward_batch.input_ids.device
        dtype = next(self.model.backbone.parameters()).dtype
        rows: list[torch.Tensor] = []
        materialized: list[str] = []
        try:
            for request in requests:
                data = request.data
                state = data.state
                if data.flow_state is None:
                    data.flow_state = head.new_request(
                        prompt_latents=state.prompt_latents,
                        speaker_embedding=state.speaker_embedding,
                        n_timesteps=state.n_timesteps,
                        inference_cfg=state.inference_cfg,
                        seed=state.seed,
                    )
                self._request_data[request.request_id] = data
                materialized.append(request.request_id)

                assert (
                    len(data.req.prefix_indices) == 0
                ), "FireRedTTS3 radix prefix reuse is disabled"
                speaker_row, prompt_patch_embeds = head.prefill_embeddings(
                    data.flow_state
                )
                text_ids = data.input_ids[0, 1 : 1 + data.text_token_count].to(device)
                text_embeds = self.model.get_input_embeddings()(text_ids)
                request_rows = [
                    speaker_row.to(device=device, dtype=dtype),
                    text_embeds.to(dtype=dtype),
                    prompt_patch_embeds.to(device=device, dtype=dtype),
                ]
                # A retracted request re-prefills its generated patches, so the
                # feedback embeddings are replayed from the stored latents.
                generated = len(data.latent_patches)
                prompt_length = int(data.input_ids.numel())
                request_length = int(data.req.extend_range.length)
                assert (
                    request_length - prompt_length
                    == generated
                    == len(data.req.output_ids)
                ), "FireRedTTS3 re-prefill history must match generated patches"
                if generated:
                    replay = head.encode_patches(
                        torch.cat(data.latent_patches, dim=0).to(device)
                    )
                    request_rows.append(replay.to(dtype=dtype))
                rows.append(torch.cat(request_rows, dim=0))
            forward_batch.input_embeds = torch.cat(rows, dim=0)
        except BaseException:
            for request_id in materialized:
                data = self._request_data.pop(request_id, None)
                if data is not None:
                    self._clear_request_data(data)
            raise

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        if not requests:
            return
        hidden = self._hidden_states(result)
        if hidden.ndim == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        elif hidden.ndim != 2:
            raise RuntimeError(
                f"FireRedTTS3 expected rank-2/3 prefill hidden, got {hidden.ndim}"
            )
        head = self.model.head
        offset = 0
        last_rows: list[torch.Tensor] = []
        for request in requests:
            data = request.data
            length = int(data.req.extend_range.length)
            request_hidden = hidden[offset : offset + length]
            if request_hidden.shape[0] != length:
                raise RuntimeError("FireRedTTS3 prefill hidden rows are incomplete")
            patch_rows = data.prompt_patch_count + len(data.latent_patches)
            head.initialize_history(data.flow_state, request_hidden[-patch_rows:])
            last_rows.append(request_hidden[-1])
            offset += length
        if offset != hidden.shape[0]:
            raise RuntimeError("FireRedTTS3 prefill hidden rows do not match requests")
        self._advance(result, requests, torch.stack(last_rows), append_hidden=False)

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        if not requests:
            return
        rows: list[torch.Tensor] = []
        for request in requests:
            queue = request.data.pending_feedback_queue
            if not queue:
                raise RuntimeError("FireRedTTS3 decode is missing its latent feedback")
            rows.append(queue.popleft())
        forward_batch.input_embeds = torch.stack(rows).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.backbone.parameters()).dtype,
        )

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        if not requests:
            return
        hidden = self._hidden_states(result)
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        elif hidden.ndim != 2:
            raise RuntimeError(
                f"FireRedTTS3 expected rank-2/3 decode hidden, got {hidden.ndim}"
            )
        self._advance(result, requests, hidden, append_hidden=True)

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.LAST

    # ------------------------------------------------------------------ #
    # shared recurrence
    # ------------------------------------------------------------------ #
    def _advance(
        self,
        result: Any,
        requests: list,
        hidden: torch.Tensor,
        *,
        append_hidden: bool,
    ) -> None:
        """Run the stop head, then the flow head for requests that continue.

        Upstream evaluates the stop head before decoding a patch, so a request
        that stops contributes no latents for this step.
        """
        head = self.model.head
        scores = head.stop_scores(hidden).tolist()
        continuing_indices: list[int] = []
        for index, request in enumerate(requests):
            data = request.data
            step_index = len(data.latent_patches)
            stopped = scores[index] >= float(data.state.stop_threshold) and (
                step_index >= int(data.state.min_gen_steps)
            )
            if stopped:
                data.req.finished_reason = FINISH_MATCHED_TOKEN(data.control_token_id)
            else:
                continuing_indices.append(index)

        if continuing_indices:
            continuing = [requests[index].data for index in continuing_indices]
            steps = head.decode_batch(
                [data.flow_state for data in continuing],
                hidden[continuing_indices],
                append_hidden=append_hidden,
            )
            for data, step in zip(continuing, steps, strict=True):
                data.latent_patches.append(step.latent_patch.detach())
                data.pending_feedback_queue.append(step.feedback_embedding.detach())

        result.next_token_ids = torch.tensor(
            [request.data.control_token_id for request in requests],
            dtype=torch.long,
            device=hidden.device,
        )

    @staticmethod
    def _hidden_states(result: Any) -> torch.Tensor:
        logits_output = getattr(result, "logits_output", None)
        hidden = getattr(logits_output, "hidden_states", None)
        if hidden is None:
            hidden = getattr(result, "hidden_states", None)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError(
                "FireRedTTS3 SGLang forward did not return hidden states"
            )
        return hidden

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        self._request_data.pop(request_id, None)
        self._clear_request_data(req_data)

    def reset_request(self, request_id: str) -> None:
        req_data = self._request_data.pop(request_id, None)
        if req_data is not None:
            self._clear_request_data(req_data)

    def _clear_request_data(self, req_data: Any) -> None:
        flow_state = getattr(req_data, "flow_state", None)
        if flow_state is not None:
            self.model.head.release_request(flow_state)
        req_data.pending_feedback_queue.clear()
        req_data.flow_state = None


__all__ = ["FireRedTTS3ModelRunner"]
