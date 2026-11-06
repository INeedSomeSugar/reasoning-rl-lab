import pytest
from reasoning_rl.sampling import RolloutCacheSampler


def initial(strategy="dynamic"):
    sampler = RolloutCacheSampler(["easy", "mixed", "hard"], strategy, 13,
                                  max_trials=8, ttl_steps=3)
    sampler.initialize_ordered({"easy": [1, 1, 1, 1], "mixed": [0, 1, 0, 1], "hard": [0, 0, 0, 0]})
    return sampler


def test_cached_dynamic_and_static_have_identical_initial_distribution():
    dynamic, static = initial(), initial("static")
    assert dynamic.weights() == static.weights()
    assert dynamic.draw(20) == static.draw(20)
    frozen = static.weights()
    dynamic.observe_rollouts("easy", [0, 0, 0, 0], 1)
    assert dynamic.weights() != frozen
    static.advance(20)
    assert static.weights() == frozen
    with pytest.raises(ValueError, match="Static"):
        static.observe_rollouts("easy", [0, 0], 21)


def test_fifo_uses_actual_probe_order_and_no_more_than_capacity():
    sampler = initial()
    sampler.observe_rollouts("mixed", [1, 1, 0, 0], 1)
    sampler.observe_rollouts("mixed", [0, 0, 0, 1], 2)
    assert sampler.cache["mixed"] == [1, 1, 0, 0, 0, 0, 0, 1]
    assert sampler.probabilities["mixed"] == pytest.approx(3.5/9)


def test_expiry_reverts_to_prior_then_restarts_without_stale_observations():
    sampler = initial()
    sampler.observe_rollouts("mixed", [0, 1, 0, 1], 1)
    sampler.advance(3)
    assert sampler.probabilities["easy"] == sampler.probabilities["hard"] == 0.5
    assert sampler.cache["easy"] == []
    sampler.observe_rollouts("easy", [1, 1, 1, 1], 3)
    assert sampler.cache["easy"] == [1, 1, 1, 1]
    assert "easy" not in sampler.expired
    assert min(sampler.weights()) >= 0.25/3
    assert sum(sampler.weights()) == pytest.approx(1)
    with pytest.raises(ValueError, match="monotone"):
        sampler.advance(2)


def test_cache_rejects_out_of_pool_and_nonbinary_rewards():
    sampler = initial()
    with pytest.raises(ValueError, match="train-pool"):
        sampler.observe_rollouts("validation-id", [0, 1], 1)
    with pytest.raises(ValueError, match="binary"):
        sampler.observe_rollouts("easy", [0.5], 1)
