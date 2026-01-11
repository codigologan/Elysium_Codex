import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class LoganAgent:
    def __init__(self, state_dim: int, action_dim: int, device: str = None):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.action_dim = action_dim

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(s)
            return int(q.argmax().item())

    def remember(self, s, a, r, ns, d):
        self.memory.append((np.array(s, dtype=float), int(a), float(r), np.array(ns, dtype=float), bool(d)))

    def sample(self, batch_size: int = 32):
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.vstack(s), np.array(a), np.array(r), np.vstack(ns), np.array(d)

    def __len__(self):
        return len(self.memory)
