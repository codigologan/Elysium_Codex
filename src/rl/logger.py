import csv
import os
from datetime import datetime
from collections import deque
import numpy as np


class RLCSVLogger:
    def __init__(self, path: str, window: int = 50):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.window = window
        self.rewards = deque(maxlen=window)

        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        # header includes canonical metrics + emotion vector + narration
        self.writer.writerow([
            "episode",
            "reward",
            "mean_reward_50",
            "episode_length",
            "epsilon",
            "curiosity",
            "frustration",
            "confidence",
            "fear",
            "flow",
            "narration",
            "timestamp",
        ])

    def log(self, episode: int, reward: float, length: int, epsilon: float, emotion: dict = None, narration: str = ""):
        self.rewards.append(float(reward))
        mean_r = float(np.mean(self.rewards)) if len(self.rewards) > 0 else float("nan")

        e = emotion or {}
        self.writer.writerow([
            int(episode),
            round(float(reward), 4),
            round(mean_r, 4),
            int(length),
            round(float(epsilon), 4),
            round(float(e.get("curiosity", 0.0)), 4),
            round(float(e.get("frustration", 0.0)), 4),
            round(float(e.get("confidence", 0.0)), 4),
            round(float(e.get("fear", 0.0)), 4),
            round(float(e.get("flow", 0.0)), 4),
            str(narration),
            datetime.utcnow().isoformat(),
        ])
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass
