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
