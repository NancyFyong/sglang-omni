# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fireredtts3.flow_head import FireRedTTS3LatentHead
from sglang_omni.models.fireredtts3.hf_config import FireRedTTS3Config

REDAE_DIM = 8
SPK_DIM = 6
PATCH_SIZE = 2
HIDDEN = 16


def _head() -> FireRedTTS3LatentHead:
    config = FireRedTTS3Config(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        vocab_size=64,
        redae_dim=REDAE_DIM,
        spk_in_dim=SPK_DIM,
        patch_size=PATCH_SIZE,
        num_history_patches=2,
        patch_encoder_hidden_size=16,
        patch_encoder_depth=1,
        patch_encoder_num_heads=2,
        patch_encoder_mlp_ratio=2,
        dit_hidden_size=16,
        dit_depth=1,
        dit_num_heads=2,
        dit_mlp_ratio=2,
    )
    torch.manual_seed(0)
    return FireRedTTS3LatentHead(config).eval()


def _state(head: FireRedTTS3LatentHead, *, seed: int | None = 11, frames: int = 6):
    # Deterministic inputs so repeated calls build byte-identical states.
    generator = torch.Generator().manual_seed(1234)
    return head.new_request(
        prompt_latents=torch.randn(1, frames, REDAE_DIM, generator=generator),
        speaker_embedding=torch.randn(1, SPK_DIM, generator=generator),
        n_timesteps=3,
        inference_cfg=2.0,
        seed=seed,
    )


def test_head_parameters_stay_float32() -> None:
    head = _head()

    assert {parameter.dtype for parameter in head.parameters()} == {torch.float32}


def test_new_request_keeps_the_dummy_history_and_prompt_frames() -> None:
    head = _head()

    state = _state(head)

    assert state.prompt_frames == 6
    assert state.latents.shape == (1, head.history_length + 6, REDAE_DIM)
    assert torch.count_nonzero(state.latents[:, : head.history_length]) == 0
    assert state.cond_history.shape == (1, head.history_patches, HIDDEN)


def test_prefill_embeddings_emit_one_row_per_patch() -> None:
    head = _head()
    state = _state(head, frames=8)

    speaker_row, prompt_patch_embeds = head.prefill_embeddings(state)

    assert speaker_row.shape == (1, HIDDEN)
    assert prompt_patch_embeds.shape == (8 // PATCH_SIZE, HIDDEN)


def test_new_request_rejects_unaligned_prompt_latents() -> None:
    head = _head()

    with pytest.raises(ValueError, match="multiple of patch_size"):
        head.new_request(
            prompt_latents=torch.randn(1, 5, REDAE_DIM),
            speaker_embedding=torch.randn(1, SPK_DIM),
            n_timesteps=3,
            inference_cfg=2.0,
            seed=1,
        )


def test_initialize_history_rebuilds_the_zero_prefix_for_a_reprefill() -> None:
    head = _head()
    state = _state(head)
    rows = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN)

    head.initialize_history(state, rows)
    head.initialize_history(state, rows)

    # Rebuilding is idempotent: a retracted request must not stack histories.
    assert state.cond_history.shape == (1, head.history_patches + 3, HIDDEN)
    assert torch.count_nonzero(state.cond_history[:, : head.history_patches]) == 0
    assert torch.equal(state.cond_history[0, head.history_patches :], rows)


def test_decode_batch_matches_per_request_decoding() -> None:
    head = _head()
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, HIDDEN, generator=generator)
    rows = [torch.randn(3, HIDDEN, generator=generator) for _ in range(2)]

    batched_states = [_state(head, seed=index + 5) for index in range(2)]
    for state, row in zip(batched_states, rows, strict=True):
        head.initialize_history(state, row)
    batched = head.decode_batch(batched_states, hidden, append_hidden=True)

    singles = []
    for index in range(2):
        state = _state(head, seed=index + 5)
        head.initialize_history(state, rows[index])
        singles.append(
            head.decode_batch([state], hidden[index : index + 1], append_hidden=True)[0]
        )

    for one, other in zip(batched, singles, strict=True):
        torch.testing.assert_close(one.latent_patch, other.latent_patch)
        torch.testing.assert_close(one.feedback_embedding, other.feedback_embedding)


def test_decode_batch_groups_mixed_timestep_requests() -> None:
    head = _head()
    states = [_state(head, seed=1), _state(head, seed=1)]
    states[1].n_timesteps = 5
    for state in states:
        head.initialize_history(state, torch.zeros(3, HIDDEN))

    steps = head.decode_batch(states, torch.zeros(2, HIDDEN), append_hidden=False)

    # Same seed and conditioning, different solver budgets => different patches.
    assert not torch.equal(steps[0].latent_patch, steps[1].latent_patch)


def test_decode_batch_advances_state_and_shapes() -> None:
    head = _head()
    state = _state(head)
    head.initialize_history(state, torch.zeros(3, HIDDEN))
    before = state.latents.shape[1]

    step = head.decode_batch([state], torch.zeros(1, HIDDEN), append_hidden=True)[0]

    assert step.latent_patch.shape == (1, PATCH_SIZE, REDAE_DIM)
    assert step.feedback_embedding.shape == (HIDDEN,)
    assert state.latents.shape[1] == before + PATCH_SIZE
    assert state.patch_count == 1
    assert state.cond_history.shape[1] == head.history_patches + 4


def test_the_recurrence_retains_no_autograd_graph() -> None:
    """The rollout must not accumulate a graph through the request state.

    ``state.latents`` and ``state.cond_history`` are re-``cat``-ed on every
    step, so a live graph would be retained for the whole generation and grow
    until the GPU is exhausted under concurrency.
    """
    head = _head()

    assert not any(param.requires_grad for param in head.parameters())
    assert head.training is False

    state = _state(head)
    head.initialize_history(state, torch.zeros(3, HIDDEN))
    for _ in range(3):
        step = head.decode_batch([state], torch.zeros(1, HIDDEN), append_hidden=True)[0]
        assert step.latent_patch.grad_fn is None
        assert step.feedback_embedding.grad_fn is None

    assert state.latents.grad_fn is None
    assert state.cond_history.grad_fn is None


def test_seed_makes_the_flow_step_reproducible() -> None:
    head = _head()
    hidden = torch.zeros(1, HIDDEN)

    def _decode(seed: int | None) -> torch.Tensor:
        state = _state(head, seed=seed)
        head.initialize_history(state, torch.zeros(3, HIDDEN))
        return head.decode_batch([state], hidden, append_hidden=False)[0].latent_patch

    assert torch.equal(_decode(21), _decode(21))
    assert not torch.equal(_decode(21), _decode(22))
    # Without a seed the flow head draws from the global RNG.
    torch.manual_seed(0)
    first = _decode(None)
    torch.manual_seed(0)
    assert torch.equal(first, _decode(None))


def test_stop_scores_are_probabilities() -> None:
    head = _head()

    scores = head.stop_scores(torch.randn(4, HIDDEN))

    assert scores.shape == (4,)
    assert bool(((scores >= 0) & (scores <= 1)).all())
