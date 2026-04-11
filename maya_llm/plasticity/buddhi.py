"""
Buddhi â€” S-Curve Consolidation Gate for LoRA Continual Fine-Tuning
Nexus Learning Labs â€” Maya Research Series

Prevents premature protection of unverified domain adaptations.
Buddhi score accumulates with training experience per domain.
The S-curve is architecturally deterministic â€” a confirmed series constant (P4â€“P9).
"""

import math
from maya_llm.utils.config import (
    BUDDHI_STEEPNESS,
    BUDDHI_MIDPOINT,
    BUDDHI_MIN_EXPERIENCE,
)


class BuddhiGate:
    def __init__(self, steps_per_domain: int):
        self.steps_per_domain = steps_per_domain
        self.experience = 0.0   # normalised [0, 1]
        self._step_count = 0
        self.score = 0.0        # S-curve output

    def step(self) -> None:
        self._step_count += 1
        self.experience = min(1.0, self._step_count / max(1, self.steps_per_domain))
        self.score = self._s_curve(self.experience)

    def _s_curve(self, x: float) -> float:
        """Sigmoid S-curve â€” series constant shape from P4."""
        return 1.0 / (1.0 + math.exp(-BUDDHI_STEEPNESS * (x - BUDDHI_MIDPOINT)))

    def is_open(self) -> bool:
        """Gate only opens after minimum experience threshold."""
        return self._step_count >= BUDDHI_MIN_EXPERIENCE

    def effective_protection_threshold(self, base_threshold: float) -> float:
        """
        Buddhi modulates Vairagya protection threshold.
        Young model (low Buddhi) â†’ conservative protection (high threshold).
        Mature model (high Buddhi) â†’ full protection (base threshold active).
        Mirrors developmental arc from Maya-Shunyata (P8).
        """
        if not self.is_open():
            return 1.0   # no consolidation before minimum experience
        return base_threshold + (1.0 - base_threshold) * (1.0 - self.score)

    def reset_for_domain(self) -> None:
        self._step_count = 0
        self.experience = 0.0
        self.score = 0.0

    def state_dict(self) -> dict:
        return {
            "score": self.score,
            "experience": self.experience,
            "step_count": self._step_count,
            "gate_open": self.is_open(),
        }


