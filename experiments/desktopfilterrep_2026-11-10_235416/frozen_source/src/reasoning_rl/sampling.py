"""An interpretable curriculum baseline, not a claim of a novel algorithm."""
import math
import random
from dataclasses import dataclass, field


@dataclass
class DifficultySampler:
    ids: list[str]
    strategy: str
    seed: int
    uniform_mix: float = 0.25
    prior: float = 0.5
    probabilities: dict[str, float] = field(default_factory=dict)
    updates: int = 0

    def __post_init__(self):
        if self.strategy not in {"uniform", "dynamic", "static"}:
            raise ValueError(self.strategy)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Non-empty unique train IDs required")
        if not 0 < self.uniform_mix <= 1 or self.prior <= 0:
            raise ValueError("Positive uniform coverage and prior required")
        self.rng = random.Random(self.seed)
        self.counts = {i: 0 for i in self.ids}

    def refresh(self, observations: dict[str, tuple[int, int]]):
        if self.strategy == "uniform":
            raise ValueError("Uniform sampler does not use difficulty probes")
        if self.strategy == "static" and self.updates:
            raise ValueError("Static ablation forbids dynamic refresh")
        if set(observations) != set(self.ids):
            raise ValueError("Probe every train-pool item; no validation or holdout items")
        for i, (successes, trials) in observations.items():
            if not 0 <= successes <= trials or trials < 2:
                raise ValueError("At least two valid probe attempts per question")
            self.probabilities[i] = (successes+self.prior)/(trials+2*self.prior)
        self.updates += 1

    def weights(self):
        n = len(self.ids)
        if self.strategy == "uniform":
            return [1/n]*n
        if set(self.probabilities) != set(self.ids):
            raise ValueError("Initial difficulty probe required")
        scores = [4*self.probabilities[i]*(1-self.probabilities[i]) for i in self.ids]
        total = sum(scores)
        return [self.uniform_mix/n+(1-self.uniform_mix)*s/total for s in scores]

    def draw(self, n):
        chosen = self.rng.choices(self.ids, weights=self.weights(), k=n)
        for i in chosen:
            self.counts[i] += 1
        return chosen

    def state_dict(self):
        w = self.weights()
        return {"strategy": self.strategy, "seed": self.seed, "uniform_mix": self.uniform_mix,
                "prior": self.prior, "updates": self.updates, "p_success": self.probabilities,
                "selection_counts": self.counts, "weights": dict(zip(self.ids, w, strict=True)),
                "entropy": -sum(x*math.log(x) for x in w), "effective_pool_size": 1/sum(x*x for x in w)}


class RolloutCacheSampler(DifficultySampler):
    """One initial full probe, then only already-paid training rewards.

    FIFO binary observations are capped per question. Dynamic estimates older than
    ttl optimizer steps revert to the symmetric prior at selection time. Static
    keeps the EXACT initial distribution, including disabling expiry.
    """
    def __init__(self, ids, strategy, seed, uniform_mix=0.25, prior=0.5,
                 max_trials=16, ttl_steps=32):
        super().__init__(ids, strategy, seed, uniform_mix, prior)
        if strategy not in {"dynamic", "static"} or max_trials < 2 or ttl_steps < 1:
            raise ValueError("Cache requires dynamic/static, >=2 trials and positive TTL")
        self.max_trials, self.ttl_steps = max_trials, ttl_steps
        self.cache = {i: [] for i in ids}
        self.last_observed = {i: None for i in ids}
        self.step = 0
        self.rollout_update_events = 0
        self.expiry_events = 0
        self.expired = set()

    def refresh(self, observations):
        super().refresh(observations)
        for i, (successes, trials) in observations.items():
            if trials > self.max_trials:
                raise ValueError("Initial probe exceeds cache capacity")
            # Aggregate initialization is an unordered block; later FIFO truncation
            # drops that block in a deterministic order, not a fictitious new trial.
            self.cache[i] = [1.0] * int(successes) + [0.0] * (trials-int(successes))
            self.last_observed[i] = 0

    def initialize_ordered(self, rewards_by_id):
        """Preserve actual initial-probe ordering for FIFO expiry."""
        if set(rewards_by_id) != set(self.ids):
            raise ValueError("Complete train-pool probe required")
        if any(v not in {0, 1, 0.0, 1.0} for values in rewards_by_id.values() for v in values):
            raise ValueError("Initial probe rewards must be binary")
        self.refresh({i: (sum(values), len(values)) for i, values in rewards_by_id.items()})
        self.cache = {i: list(values) for i, values in rewards_by_id.items()}

    def advance(self, step):
        if step < self.step:
            raise ValueError("Cache time must be monotone")
        self.step = step
        if self.strategy == "static":
            return
        for i in self.ids:
            if self.last_observed[i] is not None and step-self.last_observed[i] >= self.ttl_steps:
                if i not in self.expired:
                    self.expiry_events += 1
                self.expired.add(i)
                self.cache[i] = []
                self.probabilities[i] = 0.5

    def observe_rollouts(self, identifier, rewards, step):
        if self.strategy != "dynamic":
            raise ValueError("Static weights cannot consume later observations")
        if identifier not in self.cache or not rewards or any(v not in {0, 1, 0.0, 1.0} for v in rewards):
            raise ValueError("Only binary train-pool rewards may update the cache")
        self.advance(step)
        self.expired.discard(identifier)
        self.cache[identifier] = (self.cache[identifier] + list(rewards))[-self.max_trials:]
        self.last_observed[identifier] = step
        values = self.cache[identifier]
        self.probabilities[identifier] = (sum(values)+self.prior)/(len(values)+2*self.prior)
        self.rollout_update_events += 1
        self.updates += 1

    def state_dict(self):
        return {**super().state_dict(), "mechanism": "rollout_cache_fifo_v1",
                "cache_max_trials": self.max_trials, "cache_ttl_optimizer_steps": self.ttl_steps,
                "optimizer_step": self.step, "cache": self.cache,
                "last_observed_step": self.last_observed,
                "stale_reverted_to_prior_ids": sorted(self.expired),
                "rollout_update_events": self.rollout_update_events, "expiry_events": self.expiry_events,
                "complete_full_pool_probes": 1 if self.updates else 0,
                "post_initial_generation_probes": 0}
