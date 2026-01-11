import os
import json
import csv
import time
import random
from collections import deque
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.rl.envs.gridworld import GridWorld
from src.rl.agent import LoganAgent
from src.rl.logger import RLCSVLogger
from src.rl.dream import EpisodeReplay, DreamRunner, DreamConfig
from src.rl.dream_logger import DreamCSVLogger
from src.utils import ensure_dir, load_json, save_json, seed_everything


def append_dream_csv(
    run_dir: str,
    night: int,
    loss: float,
    novelty_rate: float,
    mix_prob: float,
    sigma: float,
    steps: int,
    episode: int,
) -> None:
    path = os.path.join(run_dir, "dream_history.csv")
    exists = os.path.exists(path)
    from datetime import datetime

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["night", "loss", "novelty_rate", "mix_prob", "sigma", "steps", "episode", "timestamp"])
        w.writerow([night, loss, novelty_rate, mix_prob, sigma, steps, episode, datetime.now().isoformat()])


def train_loop(
    config_path: str = None,
    episodes: int = 200,
    max_steps: int = 100,
    gamma: float = 0.99,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = None,
    dream_every: int = 25,
    dream_steps: int = 800,
    dream_batch: int = 128,
    dream_sigma: float = 0.02,
    dream_mix_prob: float = 0.20,
):
    # load config if provided
    cfg = {}
    if config_path and os.path.exists(config_path):
        cfg = load_json(config_path)

    run_name = cfg.get("run_name", "rl_logan_v1")
    episodes = int(cfg.get("episodes", episodes))
    max_steps = int(cfg.get("max_steps", max_steps))
    batch_size = int(cfg.get("batch_size", batch_size))
    lr = float(cfg.get("lr", lr))
    gamma = float(cfg.get("gamma", gamma))
    device = cfg.get("device", device)
    seed = int(cfg.get("seed", 42))
    dream_every = int(cfg.get("dream_every", dream_every))
    dream_steps = int(cfg.get("dream_steps", dream_steps))
    dream_batch = int(cfg.get("dream_batch", dream_batch))
    dream_sigma = float(cfg.get("dream_sigma", dream_sigma))
    dream_mix_prob = float(cfg.get("dream_mix_prob", dream_mix_prob))
    dream_enabled = bool(cfg.get("dream_enabled", True))

    # seed + dirs
    seed_everything(seed)
    base_run_dir = os.path.join("runs", run_name)
    models_base_dir = os.path.join("models", run_name)

    # ensure unique run directory to avoid overwriting previous runs
    run_dir = base_run_dir
    models_dir = models_base_dir
    if os.path.exists(run_dir):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"{base_run_dir}_{ts}"
        models_dir = f"{models_base_dir}_{ts}"

    ensure_dir(run_dir)
    ensure_dir(models_dir)
    ensure_dir("reports")

    # persist config (store effective run_name)
    try:
        cfg_out = cfg.copy() if isinstance(cfg, dict) else {}
        cfg_out.setdefault("run_name", run_name)
        cfg_out["effective_run_dir"] = run_dir
        save_json(os.path.join(run_dir, "config.json"), cfg_out)
    except Exception:
        pass

    env = GridWorld(size=int(cfg.get("size", 5)))
    agent = LoganAgent(state_dim=4, action_dim=4, device=device)
    opt = optim.Adam(agent.model.parameters(), lr=lr)
    target_model = copy.deepcopy(agent.model)
    loss_fn = nn.MSELoss()

    # reward shaping params
    use_shaping = bool(cfg.get("use_shaping", True))
    revisit_penalty = float(cfg.get("revisit_penalty", -0.2))
    time_bonus_scale = float(cfg.get("time_bonus_scale", 0.5))

    # Emotion system
    from src.rl.emotion import EmotionState
    emotion = EmotionState(window=int(cfg.get("mean_window", 50)))
    consecutive_negative = 0
    recent_rewards = []
    base_lr = lr

    writer = SummaryWriter(run_dir)

    # RL history logger (per-run canonical file)
    history_path = os.path.join(run_dir, "rl_history.csv")
    logger = RLCSVLogger(history_path, window=int(cfg.get("mean_window", 50)))

    ep_replay = EpisodeReplay(capacity=int(cfg.get("dream_replay_capacity", 500)))
    dream_cfg = DreamConfig(
        enabled=dream_enabled and dream_every > 0,
        every_episodes=dream_every,
        steps=dream_steps,
        batch_size=dream_batch,
        sigma=dream_sigma,
        mix_prob=dream_mix_prob,
        gamma=gamma,
    )
    dreamer = DreamRunner(
        online_net=agent.model,
        target_net=target_model,
        optimizer=opt,
        device=agent.device,
        n_actions=4,
        cfg=dream_cfg,
    )
    dream_csv = DreamCSVLogger(os.path.join("reports", "rl_dreams.csv"))
    night = 0

    rewards = []
    best_mean = -1e9

    for ep in range(episodes):
        eps_start = float(cfg.get("eps_start", 1.0))
        eps_end = float(cfg.get("eps_end", 0.05))
        eps_decay = float(cfg.get("eps_decay_episodes", 150))
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (ep / max(1.0, eps_decay)))
        agent.epsilon = float(epsilon)

        state = env.reset()
        # track visited grid positions to penalize revisits
        visited = set()
        # add starting position
        try:
            visited.add(tuple(env.pos.tolist()))
        except Exception:
            pass
        ep_reward = 0.0
        states = []
        actions = []
        rewards_ep = []
        next_states = []
        dones = []

        for t in range(max_steps):
            action = agent.act(state)
            # capture prev pos then step
            try:
                prev_pos = tuple(env.pos.tolist())
            except Exception:
                prev_pos = None
            next_state, reward, done = env.step(action)

            # apply reward shaping
            shaped_reward = float(reward)
            if use_shaping:
                try:
                    new_pos = tuple(env.pos.tolist())
                    if new_pos in visited:
                        shaped_reward += revisit_penalty
                    else:
                        visited.add(new_pos)
                except Exception:
                    pass

            if done:
                # bonus for reaching goal earlier in the episode
                shaped_reward += float(time_bonus_scale) * (1.0 - (t / max_steps))

            states.append(state)
            actions.append(action)
            rewards_ep.append(shaped_reward)
            next_states.append(next_state)
            dones.append(1.0 if done else 0.0)

            agent.remember(state, action, shaped_reward, next_state, done)
            ep_reward += shaped_reward
            state = next_state

            # learning step
            if len(agent) >= batch_size:
                s_b, a_b, r_b, ns_b, d_b = agent.sample(batch_size)
                s_t = torch.tensor(s_b, dtype=torch.float32, device=agent.device)
                ns_t = torch.tensor(ns_b, dtype=torch.float32, device=agent.device)
                a_t = torch.tensor(a_b, dtype=torch.int64, device=agent.device)
                r_t = torch.tensor(r_b, dtype=torch.float32, device=agent.device)
                d_t = torch.tensor(d_b, dtype=torch.float32, device=agent.device)

                q_vals = agent.model(s_t)
                q_a = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = agent.model(ns_t).max(1)[0]
                    target = r_t + (1.0 - d_t) * gamma * q_next

                loss = loss_fn(q_a, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if done:
                break

        # update emotion state
        rewards.append(ep_reward)
        recent_rewards.append(ep_reward)
        if len(recent_rewards) > emotion.window:
            recent_rewards = recent_rewards[-emotion.window:]

        # compute running mean reward
        mean_r = float(sum(rewards[-emotion.window:]) / max(1, min(len(rewards), emotion.window)))

        # update emotion from last episode
        if ep_reward < 0:
            consecutive_negative += 1
        else:
            consecutive_negative = 0

        try:
            emotion.update(reward=ep_reward, mean_reward=mean_r, epsilon=agent.epsilon, consecutive_negative=consecutive_negative, recent_rewards=list(recent_rewards))
        except Exception:
            pass

        # apply modulation
        mod = emotion.modulation()
        # adjust epsilon conservatively
        try:
            agent.epsilon = float(max(0.01, min(1.0, agent.epsilon + mod.get("epsilon_delta", 0.0))))
        except Exception:
            pass

        # scale learning rate based on flow
        try:
            scale = float(mod.get("lr_scale", 1.0))
            for g in opt.param_groups:
                g["lr"] = base_lr * scale
        except Exception:
            pass

        writer.add_scalar("reward/episode", ep_reward, ep)
        print(f"[LoganRL] ep {ep:03d} reward {ep_reward:.2f} epsilon {agent.epsilon:.3f}")

        # log per-episode canonical history (episode length = t+1) including emotion
        try:
            episode_length = int(t + 1)
        except Exception:
            episode_length = 0
        try:
            logger.log(episode=ep, reward=ep_reward, length=episode_length, epsilon=agent.epsilon, emotion=emotion.as_dict(), narration=emotion.narrate())
        except Exception:
            pass

        # checkpointing by mean reward last 20
        window = rewards[-20:]
        mean20 = sum(window) / len(window)
        if mean20 > best_mean:
            best_mean = mean20
            torch.save({"model_state": agent.model.state_dict(), "meta": {"best_mean_reward": best_mean}}, os.path.join(models_dir, "best.pt"))
        torch.save({"model_state": agent.model.state_dict(), "meta": {"last_episode": ep}}, os.path.join(models_dir, "last.pt"))

        if states:
            states_np = np.asarray(states, dtype=np.float32)
            next_states_np = np.asarray(next_states, dtype=np.float32)

            states_t = torch.from_numpy(states_np)
            next_states_t = torch.from_numpy(next_states_np)
            actions_t = torch.as_tensor(actions, dtype=torch.long)
            rewards_t = torch.as_tensor(rewards_ep, dtype=torch.float32)
            dones_t = torch.as_tensor(dones, dtype=torch.float32)

            ep_replay.add_episode({
                "states": states_t,
                "actions": actions_t,
                "rewards": rewards_t,
                "next_states": next_states_t,
                "dones": dones_t,
                "return_sum": float(ep_reward),
            })

        if dream_cfg.enabled and dream_cfg.every_episodes > 0 and (ep + 1) % dream_cfg.every_episodes == 0:
            night += 1
            try:
                target_model.load_state_dict(agent.model.state_dict())
            except Exception:
                pass
            metrics = dreamer.run_night(ep_replay)
            dream_csv.append(run_name=run_name, night=night, metrics=metrics)
            writer.add_scalar("dream/loss", metrics["dream_loss"], night)
            writer.add_scalar("dream/novelty_rate", metrics["novelty_rate"], night)
            writer.add_scalar("dream/sigma", metrics["distortion_sigma"], night)
            writer.add_scalar("dream/mix_prob", metrics["mix_prob"], night)
            writer.add_scalar("dream/steps", metrics["dream_steps"], night)
            try:
                append_dream_csv(
                    run_dir=run_dir,
                    night=night,
                    loss=float(metrics["dream_loss"]),
                    novelty_rate=float(metrics["novelty_rate"]),
                    mix_prob=float(metrics["mix_prob"]),
                    sigma=float(metrics["distortion_sigma"]),
                    steps=int(metrics["dream_steps"]),
                    episode=int(ep),
                )
            except Exception:
                pass
            print(
                f"[LoganRL][DREAM] night={night} loss={metrics['dream_loss']:.4f} "
                f"novelty={metrics['novelty_rate']:.3f}"
            )

    # save aggregated result CSV compatible with leaderboard (one row per run)
    from datetime import datetime

    mean_last_50 = float(np.mean(rewards[-50:])) if len(rewards) >= 1 else float('nan')

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "episodes": len(rewards),
        "mean_reward_last_50": mean_last_50,
        "max_reward": float(max(rewards)) if rewards else float('nan'),
        "min_reward": float(min(rewards)) if rewards else float('nan'),
    }

    csv_path = os.path.join("reports", "rl_results.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "run_name",
                "episodes",
                "mean_reward_last_50",
                "max_reward",
                "min_reward",
            ],
        )
        if write_header:
            w.writeheader()
        w.writerow(row)
    # close logger and writer
    try:
        logger.close()
    except Exception:
        pass

    writer.close()
    print("[LoganRL] treino finalizado")


def main():
    # fallback to config file if present
    cfg_path = os.path.join("configs", "rl_dqn.json")
    train_loop(config_path=cfg_path)


if __name__ == "__main__":
    main()

