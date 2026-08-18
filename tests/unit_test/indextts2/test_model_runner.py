# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.indextts2.model_runner import IndexTTS2ModelRunner

HIDDEN = 8
START_MEL = 8192


class _StubModel(torch.nn.Module):
    """Records the embedding assembly the runner is expected to perform."""

    def __init__(self) -> None:
        super().__init__()
        self.start_mel_token = START_MEL
        self.marker = torch.nn.Parameter(torch.zeros(HIDDEN))
        self.penalty: tuple[list[list[int]], list[float]] | None = None
        self.mel_calls: list[tuple[list[int], list[int]]] = []

    def condition_rows(self, speaker, emotion):
        del speaker, emotion
        return torch.zeros(3, HIDDEN)

    def text_rows(self, token_ids, language_id):
        del language_id
        return torch.ones(token_ids.numel() + 2, HIDDEN)

    def mel_rows(self, mel_ids, indices):
        self.mel_calls.append((mel_ids.tolist(), indices.tolist()))
        return torch.stack(
            [torch.full((HIDDEN,), float(value)) for value in mel_ids.tolist()]
        )

    def mel_position_index(self, offset: int) -> int:
        return 0 if offset == 0 else offset + 1

    def stage_repetition_penalty(self, token_ids, penalties) -> None:
        self.penalty = (token_ids, penalties)


def _runner(model: _StubModel) -> IndexTTS2ModelRunner:
    runner = object.__new__(IndexTTS2ModelRunner)
    runner.model = model
    return runner


def _request(request_id: str, *, generated: list[int] | None = None):
    generated = generated or []
    data = SimpleNamespace(
        state=SimpleNamespace(
            speaker_embedding=torch.zeros(1, 192),
            emotion_embedding=torch.zeros(1, 1280),
            text_token_ids=torch.tensor([[1, 2]], dtype=torch.long),
            language_id=3,
            repetition_penalty=10.0,
        ),
        pending_feedback_queue=deque(),
        req=SimpleNamespace(
            prefix_indices=[],
            output_ids=list(generated),
            extend_range=SimpleNamespace(length=3 + 4 + 1 + len(generated)),
        ),
    )
    return SimpleNamespace(request_id=request_id, data=data)


def test_prefill_stages_condition_text_and_start_rows() -> None:
    model = _StubModel()
    runner = _runner(model)
    requests = [_request("a"), _request("b")]
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(16, dtype=torch.long), input_embeds=None
    )

    runner.before_prefill(forward_batch, None, requests)

    # 3 condition rows + 4 text rows + the mel start row, per request.
    assert forward_batch.input_embeds.shape == (2 * 8, HIDDEN)
    assert model.mel_calls[0] == ([START_MEL], [0])


def test_prefill_replays_generated_codes_after_a_retraction() -> None:
    model = _StubModel()
    runner = _runner(model)
    request = _request("a", generated=[100, 101])
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(10, dtype=torch.long), input_embeds=None
    )

    runner.before_prefill(forward_batch, None, [request])

    # Upstream reads the mel position off the attention-mask width, so the start
    # row uses index 0 and the first sampled code uses index 2.
    assert model.mel_calls[0] == ([START_MEL, 100, 101], [0, 2, 3])
    assert forward_batch.input_embeds.shape == (3 + 4 + 3, HIDDEN)


def test_prefill_rejects_a_row_count_mismatch() -> None:
    model = _StubModel()
    runner = _runner(model)
    request = _request("a")
    request.data.req.extend_range = SimpleNamespace(length=99)
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(99, dtype=torch.long), input_embeds=None
    )

    with pytest.raises(RuntimeError, match="staged 8 prefill rows"):
        runner.before_prefill(forward_batch, None, [request])


def test_decode_embeds_the_last_sampled_code() -> None:
    model = _StubModel()
    runner = _runner(model)
    requests = [_request("a", generated=[100]), _request("b", generated=[200, 201])]
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(2, dtype=torch.long), input_embeds=None
    )

    runner.before_decode(forward_batch, None, requests)

    assert model.mel_calls[-1] == ([100, 201], [2, 3])
    assert forward_batch.input_embeds.shape == (2, HIDDEN)


def test_decode_requires_a_sampled_code() -> None:
    model = _StubModel()
    runner = _runner(model)
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(1, dtype=torch.long), input_embeds=None
    )

    with pytest.raises(RuntimeError, match="decode ran before the first sample"):
        runner.before_decode(forward_batch, None, [_request("a")])


def test_penalty_context_covers_the_upstream_token_set() -> None:
    model = _StubModel()
    runner = _runner(model)
    request = _request("a", generated=[100, 101])
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(2, dtype=torch.long), input_embeds=None
    )

    runner.before_decode(forward_batch, None, [request])

    token_ids, penalties = model.penalty
    # Upstream fills the prefix with id 1 and its HF penalty sees those rows
    # plus the mel start token.
    assert token_ids == [[1, START_MEL, 100, 101]]
    assert penalties == [10.0]
