import numpy as np


class GridWorld:
    """Small deterministic GridWorld.

    State: concatenation of agent pos and goal pos normalized by size (4 floats).
    Actions: 0=up,1=down,2=right,3=left
    Rewards: +1 on goal, -0.01 per step, -1 for hitting wall (stay in place).
    """

    def __init__(self, size: int = 5):
        self.size = int(size)
        self.reset()

    def reset(self):
        self.pos = np.array([0, 0], dtype=int)
        self.goal = np.array([self.size - 1, self.size - 1], dtype=int)
        return self._state()

    def _state(self):
        return np.concatenate([self.pos.astype(float) / self.size, self.goal.astype(float) / self.size]).astype(float)

    def step(self, action: int):
        move = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([1, 0]),
            3: np.array([-1, 0]),
        }[int(action)]

        new_pos = self.pos + move
        reward = -0.01
        done = False

        if np.any(new_pos < 0) or np.any(new_pos >= self.size):
            reward = -1.0
            # stay in place
        else:
            self.pos = new_pos

        if np.all(self.pos == self.goal):
            reward = 1.0
            done = True

        return self._state(), float(reward), bool(done)
