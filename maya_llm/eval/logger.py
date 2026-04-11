"""
Structured Batch Logger for Maya-LLM
Nexus Learning Labs â€” Maya Research Series

Logs all affective dimensions per step to CSV.
Matches Maya series convention â€” enables pre-experiment protocol analysis.
"""

import csv
import os
from datetime import datetime
from maya_llm.utils.config import LOG_DIR, CANARY_STRING


class MayaLLMLogger:
    def __init__(self, run_name: str):
        print(f"[Logger] Canary: {CANARY_STRING}")
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOG_DIR, f"{run_name}_{timestamp}.csv")
        self.boundary_log_path = os.path.join(LOG_DIR, f"{run_name}_{timestamp}_boundary.csv")
        self._step_writer = None
        self._boundary_writer = None
        self._step_file = None
        self._boundary_file = None
        self._init_files()

    def _init_files(self) -> None:
        self._step_file = open(self.log_path, "w", newline="")
        step_fields = [
            "step", "domain", "loss",
            "bhaya_fired", "bhaya_firing_rate",
            "buddhi_score", "vairagya_protection_frac",
            "karma_pruned_frac", "prana_level",
        ]
        self._step_writer = csv.DictWriter(self._step_file, fieldnames=step_fields)
        self._step_writer.writeheader()

        self._boundary_file = open(self.boundary_log_path, "w", newline="")
        boundary_fields = [
            "domain", "karma_pruned_weights", "prana_post_boundary",
            "vairagya_protection_frac", "buddhi_score_at_boundary",
            "AA", "BWT", "FWT",
        ]
        self._boundary_writer = csv.DictWriter(self._boundary_file, fieldnames=boundary_fields)
        self._boundary_writer.writeheader()

    def log_step(self, step: int, domain: int, loss: float, affective: dict) -> None:
        row = {"step": step, "domain": domain, "loss": round(loss, 6), **affective}
        self._step_writer.writerow(row)

    def log_boundary(self, boundary_state: dict, metrics: dict) -> None:
        row = {**boundary_state, **metrics}
        self._boundary_writer.writerow(row)

    def flush(self) -> None:
        if self._step_file:
            self._step_file.flush()
        if self._boundary_file:
            self._boundary_file.flush()

    def close(self) -> None:
        if self._step_file:
            self._step_file.close()
        if self._boundary_file:
            self._boundary_file.close()


