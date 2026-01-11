import os
import csv
import torch
import pandas as pd
from datetime import datetime
from src.rl.envs.gridworld import GridWorld
from src.rl.agent import LoganAgent


def evaluate(model_path: str = None, episodes: int = 100, max_steps: int = 100, device: str = None):
    env = GridWorld(size=5)
    agent = LoganAgent(state_dim=4, action_dim=4, device=device)
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=agent.device)
        agent.model.load_state_dict(ckpt.get("model_state", ckpt))

    rewards = []
    for ep in range(episodes):
        s = env.reset()
        ep_r = 0.0
        for _ in range(max_steps):
            a = agent.act(s)
            s, r, done = env.step(a)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)

    mean_reward = sum(rewards) / len(rewards)
    print(f"[LoganRL][EVAL] mean_reward={mean_reward:.4f}")

    os.makedirs("reports", exist_ok=True)
    out = os.path.join("reports", "rl_eval.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            w.writerow([i, r])
    print(f"[LoganRL][EVAL] CSV salvo: {out}")

    # integrate into main eval_results.csv: add/ensure mean_reward column and append a summary row
    main_csv = os.path.join("reports", "eval_results.csv")
    rl_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": os.path.basename(os.path.dirname(model_path)) if model_path is not None else "rl_logan_v1",
        "ckpt": os.path.basename(model_path) if model_path is not None else "best",
        "device": str(agent.device),
        "epoch": None,
        "val_loss": None,
        "val_acc": None,
        "best_val_loss_in_ckpt_meta": None,
        "mean_reward": float(mean_reward),
    }

    # if main CSV exists, load and ensure column
    if os.path.exists(main_csv):
        try:
            df = pd.read_csv(main_csv)
        except Exception:
            df = pd.DataFrame()
        if "mean_reward" not in df.columns:
            df["mean_reward"] = pd.NA
        df = pd.concat([df, pd.DataFrame([rl_row])], ignore_index=True, sort=False)
        df.to_csv(main_csv, index=False)
        print(f"[LoganRL][EVAL] integrado em: {main_csv}")
    else:
        # create new eval_results.csv with RL row
        df = pd.DataFrame([rl_row])
        df.to_csv(main_csv, index=False)
        print(f"[LoganRL][EVAL] criado: {main_csv}")


def main():
    evaluate(model_path=os.path.join("models", "rl_logan_v1", "best.pt"))


if __name__ == "__main__":
    main()
