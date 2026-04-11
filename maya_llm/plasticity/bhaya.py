"""
Bhaya â€” Nociceptive Metaplasticity for LoRA Continual Fine-Tuning
Nexus Learning Labs â€” Maya Research Series

Biological grounding: Central sensitisation (Woolf, 2011).
Loss spikes are the LLM analogue of nociceptive pain signals.
When loss spikes sharply above running mean, Bhaya fires:
  â€” elevates lability (effective LR multiplier) on active LoRA adapter weights
  â€” lability decays per step until next trigger
  â€” Vairagya-protected weights are exempt from lability elevation
"""

import torch
from collections import deque
from maya_llm.utils.config import (
    BHAYA_LOSS_SPIKE_THRESHOLD,
    BHAYA_LABILITY_MULTIPLIER,
    BHAYA_LABILITY_DECAY,
    BHAYA_WINDOW,
)


class BhayaMetaplasticity:
    def __init__(self, lora_params: list):
        self.lora_params = lora_params
        self.loss_window = deque(maxlen=BHAYA_WINDOW)
        self.lability = {id(p): 1.0 for p in lora_params}
        self.firing_rate = 0.0
        self._fire_count = 0
        self._step_count = 0

    def update(self, loss: float, vairagya_mask: dict | None = None) -> bool:
        self._step_count += 1
        self.loss_window.append(loss)

        fired = False
        if len(self.loss_window) >= 10:
            running_mean = sum(self.loss_window) / len(self.loss_window)
            if loss > BHAYA_LOSS_SPIKE_THRESHOLD * running_mean:
                self._fire(vairagya_mask)
                fired = True
                self._fire_count += 1

        self._decay_lability()
        self.firing_rate = self._fire_count / max(1, self._step_count)
        return fired

    def _fire(self, vairagya_mask: dict | None) -> None:
        for p in self.lora_params:
            pid = id(p)
            # Vairagya-protected weights are exempt â€” earned resilience overrides pain
            if vairagya_mask and vairagya_mask.get(pid, 0.0) >= 0.65:
                continue
            self.lability[pid] = BHAYA_LABILITY_MULTIPLIER

    def _decay_lability(self) -> None:
        for pid in self.lability:
            if self.lability[pid] > 1.0:
                self.lability[pid] = max(1.0, self.lability[pid] * BHAYA_LABILITY_DECAY)

    def get_lability(self, param: torch.nn.Parameter) -> float:
        return self.lability.get(id(param), 1.0)

    def is_quiescent(self) -> bool:
        return self.firing_rate == 0.0

    def state_dict(self) -> dict:
        return {
            "lability": self.lability,
            "firing_rate": self.firing_rate,
            "fire_count": self._fire_count,
            "step_count": self._step_count,
        }


