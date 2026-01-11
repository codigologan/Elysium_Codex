from dataclasses import dataclass, field
from typing import Dict


def _clamp(x: float, a: float = -1.0, b: float = 1.0) -> float:
    return max(a, min(b, float(x)))


@dataclass
class EmotionState:
    curiosity: float = 0.0
    frustration: float = 0.0
    confidence: float = 0.0
    fear: float = 0.0
    flow: float = 0.0
    window: int = 50
    history: list = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        return {
            "curiosity": round(self.curiosity, 4),
            "frustration": round(self.frustration, 4),
            "confidence": round(self.confidence, 4),
            "fear": round(self.fear, 4),
            "flow": round(self.flow, 4),
        }

    def update(self, reward: float, mean_reward: float, epsilon: float, consecutive_negative: int, recent_rewards: list, threshold: float = 0.1):
        """Update the emotion vector from episode signals.

        Args:
            reward: episode reward (float)
            mean_reward: running mean reward across runs (float)
            epsilon: current exploration rate
            consecutive_negative: number of recent consecutive negative episodes
            recent_rewards: list of recent reward floats
            threshold: sensitivity threshold for frustration
        """
        # curiosity increases with exploration and novelty (proxy: high epsilon)
        novelty_bonus = 1.0 - (len(recent_rewards) / max(1, self.window))
        self.curiosity = _clamp(self.curiosity + 0.5 * epsilon * novelty_bonus)

        # frustration increases when reward is below expectation
        if reward < (mean_reward - threshold):
            self.frustration = _clamp(self.frustration + 0.1)
        else:
            # decay frustration slowly
            self.frustration = _clamp(self.frustration * 0.95)

        # confidence grows with positive deviation
        self.confidence = _clamp(self.confidence + max(0.0, reward - mean_reward) * 0.5)

        # fear grows with repeated negative outcomes
        if consecutive_negative >= 3:
            self.fear = _clamp(self.fear + 0.2)
        else:
            self.fear = _clamp(self.fear * 0.9)

        # flow: high reward and low variance in recent window
        if recent_rewards:
            import statistics

            try:
                var = statistics.pstdev(recent_rewards)
            except Exception:
                var = float("inf")
            high_reward = reward > mean_reward
            if high_reward and var < max(1e-3, abs(mean_reward) + 1.0):
                self.flow = _clamp(self.flow + 0.2)
            else:
                self.flow = _clamp(self.flow * 0.95)

        # keep a small history for other heuristics
        self.history.append(reward)
        if len(self.history) > max(self.window, len(recent_rewards)):
            self.history = self.history[-self.window:]

    def modulation(self) -> Dict[str, float]:
        """Return modulation factors to influence agent hyperparameters.

        Returns a dict like {
            'epsilon_delta': float,
            'lr_scale': float,
            'conservatism': float
        }
        """
        # curiosity increases epsilon slightly, confidence reduces it, fear increases it
        epsilon_delta = 0.02 * (self.curiosity - self.confidence + self.fear)

        # flow reduces learning rate (stabilization), scale between 0.7-1.0
        lr_scale = max(0.7, 1.0 - 0.15 * self.flow)

        # conservatism factor (0..1) increases when fear is high
        conservatism = _clamp(self.fear)

        return {"epsilon_delta": epsilon_delta, "lr_scale": lr_scale, "conservatism": conservatism}

    def narrate(self) -> str:
        # simple prioritized narration
        if self.frustration > 0.6:
            return "sinto resistência interna. preciso mudar."
        if self.confidence > 0.7:
            return "estou confiante. posso consolidar."
        if self.flow > 0.8:
            return "estado de fluxo atingido."
        if self.fear > 0.6:
            return "ambiente instável. cautela."
        if self.curiosity > 0.6:
            return "explorando com curiosidade."
        return "explorando."
