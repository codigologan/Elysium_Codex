import argparse
import pandas as pd
import os
import glob
from datetime import datetime


def summarize_episode_file(path):
    # read per-episode CSV with columns episode,reward and compute summary
    try:
        d = pd.read_csv(path)
        if "reward" not in d.columns:
            return None
        rewards = d["reward"].astype(float).tolist()
        if not rewards:
            return None

        # prefer provided rolling mean if present (rl_history.csv), otherwise compute
        if "mean_reward_50" in d.columns:
            try:
                mean_last_50 = float(d["mean_reward_50"].iloc[-1])
            except Exception:
                mean_last_50 = float(pd.Series(rewards).tail(50).mean())
        elif "mean_reward_last_50" in d.columns:
            try:
                mean_last_50 = float(d["mean_reward_last_50"].iloc[-1])
            except Exception:
                mean_last_50 = float(pd.Series(rewards).tail(50).mean())
        else:
            mean_last_50 = float(pd.Series(rewards).tail(50).mean())

        # determine run_name: if file lives under runs/<run>/..., use parent folder name
        parent = os.path.basename(os.path.dirname(path))
        if parent:
            run_name = parent
        else:
            run_name = os.path.basename(path).split("_rl_results")[0]

        # timestamp: prefer last row timestamp column if present
        if "timestamp" in d.columns and not d["timestamp"].isna().all():
            try:
                ts = str(d["timestamp"].iloc[-1])
            except Exception:
                ts = datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
        else:
            ts = datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")

        row = {
            "timestamp": ts,
            "run_name": run_name,
            "episodes": len(rewards),
            "mean_reward_last_50": mean_last_50,
            "max_reward": float(max(rewards)),
            "min_reward": float(min(rewards)),
        }
        return row
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="reports/rl_results.csv")
    parser.add_argument("--metric", default="mean_reward_last_50")
    args = parser.parse_args()

    rows = []

    # Primary file: if it exists and already contains aggregated rows, use directly
    if os.path.exists(args.csv):
        try:
            df = pd.read_csv(args.csv)
            if "mean_reward_last_50" in df.columns:
                rows.extend(df.to_dict(orient="records"))
            elif "reward" in df.columns and "episode" in df.columns:
                s = summarize_episode_file(args.csv)
                if s:
                    rows.append(s)
        except Exception:
            pass

    # Also scan for older per-episode files like reports/*_rl_results.csv
    pattern = os.path.join("reports", "*_rl_results.csv")
    for p in glob.glob(pattern):
        if os.path.abspath(p) == os.path.abspath(args.csv):
            continue
        s = summarize_episode_file(p)
        if s:
            rows.append(s)

    if not rows:
        print(f"[LoganRL] aviso: nenhum resultado RL encontrado em {args.csv} ou reports/*_rl_results.csv")
        return

    # build dataframe and sort
    df_out = pd.DataFrame(rows)
    if args.metric not in df_out.columns:
        print(f"[LoganRL] coluna de métrica não encontrada após sumarização: {args.metric}")
        return

    df_out = df_out.sort_values(by=args.metric, ascending=False)

    print("\n[LoganRL] LEADERBOARD")
    cols = [c for c in ["timestamp", "run_name", "episodes", "mean_reward_last_50", "max_reward"] if c in df_out.columns]
    print(df_out[cols].to_string(index=False))

    out = "reports/leaderboard_rl.csv"
    df_out.to_csv(out, index=False)
    print(f"\n[LoganRL] leaderboard salvo em {out}")


if __name__ == "__main__":
    main()
