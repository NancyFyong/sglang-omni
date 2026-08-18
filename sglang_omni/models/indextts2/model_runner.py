# SPDX-License-Identifier: Apache-2.0
"""Omni ModelRunner hooks for the IndexTTS-2.5 embedding prefix.

SGLang owns sampling and stop handling because the model emits discrete mel
codes; the runner only stages the spliced input embeddings for every forward.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner


class IndexTTS2ModelRunner(ModelRunner):
    """Stage ``[condition][text][mel]`` embeddings for prefill and decode."""

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        if not requests:
            return
        model = self.model
        device = forward_batch.input_ids.device
        dtype = next(model.parameters()).dtype
        rows: list[torch.Tensor] = []
        for request in requests:
            data = request.data
            state = data.state
            assert (
                len(data.req.prefix_indices) == 0
            ), "IndexTTS-2.5 radix prefix reuse is disabled"
            request_rows = [
                model.condition_rows(state.speaker_embedding, state.emotion_embedding),
                model.text_rows(state.text_token_ids, state.language_id),
            ]
            # The start token plus, after a retraction, every code generated so
            # far are replayed so the KV cache is rebuilt identically.
            mel_ids = [model.start_mel_token, *(int(v) for v in data.req.output_ids)]
            indices = [
                model.mel_position_index(offset) for offset in range(len(mel_ids))
            ]
            request_rows.append(
                model.mel_rows(
                    torch.tensor(mel_ids, dtype=torch.long),
                    torch.tensor(indices, dtype=torch.long),
                )
            )
            request_length = int(data.req.extend_range.length)
            staged = torch.cat(request_rows, dim=0)
            if staged.shape[0] != request_length:
                raise RuntimeError(
                    f"IndexTTS-2.5 staged {staged.shape[0]} prefill rows for a "
                    f"{request_length}-token extend"
                )
            rows.append(staged.to(device=device, dtype=dtype))
        forward_batch.input_embeds = torch.cat(rows, dim=0)
        self._stage_penalty(requests)

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
        model = self.model
        mel_ids: list[int] = []
        indices: list[int] = []
        for request in requests:
            output_ids = request.data.req.output_ids
            if not output_ids:
                raise RuntimeError("IndexTTS-2.5 decode ran before the first sample")
            mel_ids.append(int(output_ids[-1]))
            # start token sits at offset 0, so the k-th sampled code is offset k.
            indices.append(model.mel_position_index(len(output_ids)))
        forward_batch.input_embeds = model.mel_rows(
            torch.tensor(mel_ids, dtype=torch.long),
            torch.tensor(indices, dtype=torch.long),
        ).to(
            device=forward_batch.input_ids.device,
            dtype=next(model.parameters()).dtype,
        )
        self._stage_penalty(requests)

    def _stage_penalty(self, requests: list) -> None:
        """Give the model the per-row token history for its repetition penalty."""
        from sglang_omni.models.indextts2.request_builders import PLACEHOLDER_TOKEN_ID

        model = self.model
        token_ids = [
            [
                PLACEHOLDER_TOKEN_ID,
                model.start_mel_token,
                *(int(value) for value in request.data.req.output_ids),
            ]
            for request in requests
        ]
        penalties = [
            float(request.data.state.repetition_penalty) for request in requests
        ]
        model.stage_repetition_penalty(token_ids, penalties)


__all__ = ["IndexTTS2ModelRunner"]
