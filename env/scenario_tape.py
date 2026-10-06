"""Order-invariant stochastic inputs for paired solver evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Sequence, Tuple


SemanticKey = Tuple[Any, ...]


@dataclass(frozen=True)
class ScenarioTape:
    """Deterministically maps semantic event keys to random variates.

    No mutable RNG state is shared across operations. A value therefore depends
    on the scenario seed and its semantic identity, not on the order in which a
    scheduling policy starts operations.
    """

    seed: int
    version: int = 1

    def uniform(
        self,
        key: SemanticKey,
        low: float,
        high: float,
    ) -> float:
        if high < low:
            raise ValueError("uniform maximum must be >= minimum")
        return self._rng(key).uniform(float(low), float(high))

    def normal(
        self,
        key: SemanticKey,
        mean: float,
        std: float,
    ) -> float:
        if std < 0.0:
            raise ValueError("normal standard deviation must be non-negative")
        if std == 0.0:
            return float(mean)
        return self._rng(key).gauss(float(mean), float(std))

    def categorical(
        self,
        key: SemanticKey,
        values: Sequence[int],
        probabilities: Sequence[float],
    ) -> int:
        if len(values) != len(probabilities):
            raise ValueError(
                "categorical values and probabilities must have equal lengths"
            )
        if not values:
            raise ValueError("categorical distribution must not be empty")
        total = sum(float(probability) for probability in probabilities)
        if total <= 0.0:
            raise ValueError(
                "categorical probabilities must sum to a positive value"
            )
        draw = self._rng(key).random() * total
        cumulative = 0.0
        for value, probability in zip(values, probabilities):
            cumulative += float(probability)
            if draw <= cumulative:
                return int(value)
        return int(values[-1])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._payload(())).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "version": int(self.version),
            "seed": int(self.seed),
            "sha256": self.sha256,
        }

    def _rng(self, key: SemanticKey) -> random.Random:
        digest = hashlib.sha256(self._payload(key)).digest()
        return random.Random(int.from_bytes(digest[:16], "big"))

    def _payload(self, key: SemanticKey) -> bytes:
        return json.dumps(
            {
                "version": int(self.version),
                "seed": int(self.seed),
                "key": list(key),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
