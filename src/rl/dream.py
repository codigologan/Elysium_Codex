from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DreamConfig:
    enabled: bool = True
    every_episodes: int = 25
    steps: int = 800
    batch_size: int = 128
    gamma: float = 0.99

    sigma: float = 0.02
    mix_prob: float = 0.20
    mix_alpha_low: float = 0.30
    mix_alpha_high: float = 0.70

    prefer_positive: bool = True
    temperature: float = 1.0

    max_grad_norm: float = 1.0


class EpisodeReplay:
    """
    Replay buffer por episodio. Cada item guarda tensors.
    Ideal para sonhos (mistura / reamostragem).
    """
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.episodes: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, episode: Dict[str, Any]) -> None:
        """
        episode esperado:
          states: Tensor [T, obs_dim]
          actions: Tensor [T] long
          rewards: Tensor [T]
          next_states: Tensor [T, obs_dim]
          dones: Tensor [T] (0/1)
          return_sum: float
        """
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def sample_transition(
        self,
        prefer_positive: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.episodes:
            raise RuntimeError("EpisodeReplay vazio")

        if prefer_positive and len(self.episodes) > 1:
            returns = np.array([ep.get("return_sum", 0.0) for ep in self.episodes], dtype=np.float64)
            z = (returns - returns.max()) / max(1e-6, temperature)
            w = np.exp(z)
            w = w / (w.sum() + 1e-12)
            ep = self.episodes[int(np.random.choice(len(self.episodes), p=w))]
        else:
            ep = random.choice(self.episodes)

        t = random.randrange(ep["actions"].shape[0])

        s = ep["states"][t]
        a = ep["actions"][t]
        r = ep["rewards"][t]
        ns = ep["next_states"][t]
        d = ep["dones"][t]
        return s, a, r, ns, d


class DreamRunner:
    """
    Executa "noites" de sonho: TD updates offline com distorcoes leves.
    Compativel com DQN classico (online_net, target_net, optimizer).
    """
    def __init__(
        self,
        online_net: nn.Module,
        target_net: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        n_actions: int,
        cfg: DreamConfig,
    ):
        self.online = online_net
        self.target = target_net if target_net is not None else online_net
        self.opt = optimizer
        self.device = device
        self.n_actions = n_actions
        self.cfg = cfg
        self.loss_fn = nn.SmoothL1Loss()

    def run_night(self, replay: EpisodeReplay) -> Dict[str, float]:
        if not self.cfg.enabled or len(replay) < 1:
            return {
                "dream_loss": float("nan"),
                "novelty_rate": 0.0,
                "distortion_sigma": float(self.cfg.sigma),
                "mix_prob": float(self.cfg.mix_prob),
                "dream_steps": 0.0,
            }

        self.online.train()
        if self.target is not self.online:
            self.target.eval()

        losses = []
        novelty_hits = 0

        for _ in range(self.cfg.steps):
            batch = [
                replay.sample_transition(self.cfg.prefer_positive, self.cfg.temperature)
                for _ in range(self.cfg.batch_size)
            ]

            s = torch.stack([b[0] for b in batch]).to(self.device)
            a = torch.stack([b[1] for b in batch]).long().to(self.device)
            r = torch.stack([b[2] for b in batch]).to(self.device)
            ns = torch.stack([b[3] for b in batch]).to(self.device)
            d = torch.stack([b[4] for b in batch]).to(self.device)

            if self.cfg.sigma > 0 or self.cfg.mix_prob > 0:
                distorted_mask = torch.zeros(s.size(0), device=s.device, dtype=torch.bool)

                if self.cfg.mix_prob > 0:
                    perm = torch.randperm(s.size(0), device=s.device)
                    s2 = s[perm]
                    ns2 = ns[perm]
                    alpha = (
                        torch.rand(s.size(0), device=s.device)
                        * (self.cfg.mix_alpha_high - self.cfg.mix_alpha_low)
                        + self.cfg.mix_alpha_low
                    ).unsqueeze(1)
                    mix_mask = (torch.rand(s.size(0), device=s.device) < self.cfg.mix_prob).unsqueeze(1)

                    s_mixed = alpha * s + (1 - alpha) * s2
                    ns_mixed = alpha * ns + (1 - alpha) * ns2

                    s = torch.where(mix_mask, s_mixed, s)
                    ns = torch.where(mix_mask, ns_mixed, ns)

                    distorted_mask |= mix_mask.squeeze(1)

                if self.cfg.sigma > 0:
                    s = s + torch.randn_like(s) * self.cfg.sigma
                    ns = ns + torch.randn_like(ns) * self.cfg.sigma
                    distorted_mask |= torch.ones(s.size(0), device=s.device, dtype=torch.bool)

                novelty_hits += int(distorted_mask.sum().item())

            with torch.no_grad():
                q_next = self.target(ns).max(dim=1).values
                y = r + (1.0 - d) * self.cfg.gamma * q_next

            q = self.online(s).gather(1, a.view(-1, 1)).squeeze(1)
            loss = self.loss_fn(q, y)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()

            if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.max_grad_norm)

            self.opt.step()

            losses.append(loss.item())

        dream_loss = float(np.mean(losses)) if losses else float("nan")
        novelty_rate = float(novelty_hits) / float(max(1, self.cfg.steps * self.cfg.batch_size))

        return {
            "dream_loss": dream_loss,
            "novelty_rate": novelty_rate,
            "distortion_sigma": float(self.cfg.sigma),
            "mix_prob": float(self.cfg.mix_prob),
            "dream_steps": float(self.cfg.steps),
        }
