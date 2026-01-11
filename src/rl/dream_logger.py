import csv
import os
from datetime import datetime


class DreamCSVLogger:
    def __init__(self, path: str = "reports/rl_dreams.csv"):
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.path = path
        self.fields = [
            "timestamp",
            "run_name",
            "night",
            "dream_steps",
            "distortion_sigma",
            "mix_prob",
            "dream_loss",
            "novelty_rate",
        ]

        self._needs_header = not os.path.exists(self.path)

    def append(self, run_name: str, night: int, metrics: dict) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "night": night,
            "dream_steps": metrics.get("dream_steps"),
            "distortion_sigma": metrics.get("distortion_sigma"),
            "mix_prob": metrics.get("mix_prob"),
            "dream_loss": metrics.get("dream_loss"),
            "novelty_rate": metrics.get("novelty_rate"),
        }

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if self._needs_header:
                w.writeheader()
                self._needs_header = False
            w.writerow(row)
