# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.fireredtts3.model_runner import FireRedTTS3ModelRunner

HIDDEN = 4
PATCH_SIZE = 2
REDAE_DIM = 3


class _StubHead:
    """Records the recurrence calls the runner is expected to make."""

    patch_size = PATCH_SIZE
    history_patches = 2

    def __init__(self, *, stop_scores: list[float]) -> None:
        self.stop_scores_value = stop_scores
        self.released: list[object] = []
        self.initialized: list[torch.Tensor] = []
        self.decode_calls: list[tuple[int, bool]] = []

    def new_request(self, **kwargs):
        return SimpleNamespace(kwargs=kwargs)

    def prefill_embeddings(self, _state):
        return torch.zeros(1, HIDDEN), torch.zeros(2, HIDDEN)

    def encode_patches(self, latents):
        return torch.zeros(latents.shape[0], HIDDEN)

    def initialize_history(self, _state, rows):
        self.initialized.append(rows)

    def stop_scores(self, hidden):
        return torch.tensor(self.stop_scores_value[: hidden.shape[0]])

    def decode_batch(self, states, hidden, *, append_hidden):
        self.decode_calls.append((len(states), append_hidden))
        return [
            SimpleNamespace(
                latent_patch=torch.full((1, PATCH_SIZE, REDAE_DIM), float(index)),
                feedback_embedding=torch.full((HIDDEN,), float(index)),
            )
            for index in range(len(states))
        ]

    def release_request(self, state):
        self.released.append(state)


def _runner(head: _StubHead) -> FireRedTTS3ModelRunner:
    runner = object.__new__(FireRedTTS3ModelRunner)
    runner.model = SimpleNamespace(
        head=head,
        backbone=nn.Linear(HIDDEN, HIDDEN),
        get_input_embeddings=lambda: nn.Embedding.from_pretrained(
            torch.zeros(16, HIDDEN)
        ),
    )
    runner._request_data = {}
    return runner


def _request(
    request_id: str,
    *,
    stop_threshold: float = 0.5,
    min_gen_steps: int = 6,
    patches: int = 0,
):
    data = SimpleNamespace(
        flow_state=None,
        prompt_patch_count=2,
        text_token_count=1,
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        latent_patches=[torch.zeros(1, PATCH_SIZE, REDAE_DIM) for _ in range(patches)],
        pending_feedback_queue=deque(),
        control_token_id=7,
        req=SimpleNamespace(
            prefix_indices=[],
            extend_range=SimpleNamespace(length=4 + patches),
            output_ids=[7] * patches,
            finished_reason=None,
        ),
        state=SimpleNamespace(
            prompt_latents=torch.zeros(1, 4, REDAE_DIM),
            speaker_embedding=torch.zeros(1, 6),
            n_timesteps=3,
            inference_cfg=2.0,
            stop_threshold=stop_threshold,
            min_gen_steps=min_gen_steps,
            seed=5,
        ),
    )
    return SimpleNamespace(request_id=request_id, data=data)


def test_prefill_stages_embeddings_in_scheduler_order() -> None:
    head = _StubHead(stop_scores=[0.0, 0.0])
    runner = _runner(head)
    requests = [_request("a"), _request("b")]
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(8, dtype=torch.long), input_embeds=None
    )

    runner.before_prefill(forward_batch, None, requests)

    # 1 speaker row + 1 text token + 2 reference patch rows, per request.
    assert forward_batch.input_embeds.shape == (8, HIDDEN)
    assert set(runner._request_data) == {"a", "b"}
    assert all(request.data.flow_state is not None for request in requests)


def test_prefill_replays_generated_patches_after_a_retraction() -> None:
    head = _StubHead(stop_scores=[0.0])
    runner = _runner(head)
    request = _request("a", patches=2)
    resumed_state = object()
    request.data.flow_state = resumed_state
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(6, dtype=torch.long), input_embeds=None
    )

    runner.before_prefill(forward_batch, None, [request])

    # The existing flow state is reused so the per-request RNG keeps its stream.
    assert request.data.flow_state is resumed_state
    assert forward_batch.input_embeds.shape == (4 + 2, HIDDEN)


def test_post_prefill_seeds_history_from_the_patch_rows() -> None:
    head = _StubHead(stop_scores=[0.0])
    runner = _runner(head)
    request = _request("a")
    request.data.flow_state = object()
    hidden = torch.arange(4 * HIDDEN, dtype=torch.float32).reshape(4, HIDDEN)
    result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden))

    runner.post_prefill(result, None, SimpleNamespace(is_prefill_only=False), [request])

    assert torch.equal(head.initialized[0], hidden[-2:])
    # The first flow step reuses the prefill row that is already in the history.
    assert head.decode_calls == [(1, False)]
    assert len(request.data.latent_patches) == 1
    assert len(request.data.pending_feedback_queue) == 1


def test_post_prefill_skips_a_prefill_only_batch() -> None:
    head = _StubHead(stop_scores=[0.0])
    runner = _runner(head)

    runner.post_prefill(
        result=object(),
        forward_batch=None,
        schedule_batch=SimpleNamespace(is_prefill_only=True),
        requests=[_request("a")],
    )

    assert head.decode_calls == []


def test_decode_consumes_one_feedback_row_per_request() -> None:
    head = _StubHead(stop_scores=[0.0, 0.0])
    runner = _runner(head)
    requests = [_request("a"), _request("b")]
    for index, request in enumerate(requests):
        request.data.pending_feedback_queue.append(torch.full((HIDDEN,), float(index)))
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(2, dtype=torch.long), input_embeds=None
    )

    runner.before_decode(forward_batch, None, requests)

    assert forward_batch.input_embeds.shape == (2, HIDDEN)
    assert forward_batch.input_embeds[1, 0] == 1.0
    assert all(not request.data.pending_feedback_queue for request in requests)


def test_stop_head_gate_respects_min_gen_steps() -> None:
    head = _StubHead(stop_scores=[0.9])
    runner = _runner(head)
    request = _request("a", min_gen_steps=2, patches=1)
    request.data.flow_state = object()
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(1, HIDDEN))
    )

    runner.post_decode(result, None, None, [request])

    # One patch generated so far, min_gen_steps=2 => the stop score is ignored.
    assert request.data.req.finished_reason is None
    assert len(request.data.latent_patches) == 2


def test_stop_head_finishes_without_emitting_another_patch() -> None:
    head = _StubHead(stop_scores=[0.9])
    runner = _runner(head)
    request = _request("a", min_gen_steps=2, patches=2)
    request.data.flow_state = object()
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(1, HIDDEN))
    )

    runner.post_decode(result, None, None, [request])

    assert request.data.req.finished_reason is not None
    assert len(request.data.latent_patches) == 2
    assert head.decode_calls == []


def test_mixed_batch_only_advances_unfinished_requests() -> None:
    head = _StubHead(stop_scores=[0.9, 0.1])
    runner = _runner(head)
    finished = _request("a", min_gen_steps=0, patches=3)
    running = _request("b", min_gen_steps=0, patches=3)
    for request in (finished, running):
        request.data.flow_state = object()
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(2, HIDDEN))
    )

    runner.post_decode(result, None, None, [finished, running])

    assert head.decode_calls == [(1, True)]
    assert finished.data.req.finished_reason is not None
    assert len(finished.data.latent_patches) == 3
    assert len(running.data.latent_patches) == 4
    assert result.next_token_ids.tolist() == [7, 7]


def test_abort_and_finish_release_the_flow_state_once() -> None:
    head = _StubHead(stop_scores=[0.0])
    runner = _runner(head)
    request = _request("a")
    flow_state = object()
    request.data.flow_state = flow_state
    request.data.pending_feedback_queue.append(torch.zeros(HIDDEN))
    runner._request_data["a"] = request.data

    runner.on_request_finished("a", request.data)
    runner.reset_request("a")

    assert runner._request_data == {}
    assert request.data.flow_state is None
    assert not request.data.pending_feedback_queue
    assert head.released == [flow_state]
