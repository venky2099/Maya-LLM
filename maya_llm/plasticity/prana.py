"""
Prana â€” Metabolic Plasticity Budget for LoRA Continual Fine-Tuning
Nexus Learning Labs â€” Maya Research Series

Biological grounding: Astrocyte-Neuron Lactate Shuttle (Pellerin & Magistretti, 1994).
Prana gates the effective learning rate per batch.
Depletes under gradient load; recovers at low-activity batches.
Vairagya-experienced parameters recover Prana more efficiently.
Never fully depletes at canonical cost rate â€” biologically accurate (P9 confirmed).
"""

import torch
from maya_llm.utils.config import (
    PRANA_INIT,
    PRANA_MIN,
    PRANA_COST_RATE,
    PRANA_RECOVERY_RATE,
    PRANA_RECOVERY_THRESHOLD,
    PRANA_BOUNDARY_RECOVERY,
)


class PranaBudget:
    def __init__(self):
        self.p = PRANA_INIT
        self._history = []

    def update(self, grad_magnitude: float, mean_activity: float, vairagya_mean: float) -> float:
        """
        Update Prana budget per batch.
        Returns effective LR multiplier for this batch.
        """
        # Depletion under gradient load
        depletion = PRANA_COST_RATE * grad_magnitude * mean_activity
        self.p = max(PRANA_MIN, self.p - depletion)

        # Recovery during low-activity batches
        if mean_activity < PRANA_RECOVERY_THRESHOLD:
            # Vairagya modulates recovery â€” experienced circuits need less fuel
            recovery = PRANA_RECOVERY_RATE * (1.0 - mean_activity) * (0.5 + vairagya_mean * 0.5)
            self.p = min(1.0, self.p + recovery)

        self._history.append(self.p)
        return self.p   # used directly as LR multiplier

    def boundary_recovery(self) -> None:
        """Partial recovery at domain boundary â€” sleep analogue."""
        self.p = min(1.0, self.p + PRANA_BOUNDARY_RECOVERY * (1.0 - self.p))

    def is_depleted(self) -> bool:
        return self.p <= PRANA_MIN + 1e-4

    def state_dict(self) -> dict:
        return {
            "prana": self.p,
            "mean_prana": sum(self._history[-100:]) / max(1, len(self._history[-100:])),
        }


