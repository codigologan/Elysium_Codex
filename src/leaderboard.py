import argparse
import pandas as pd
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="reports/eval_results.csv")
    parser.add_argument("--metric", default="val_loss", choices=["val_loss", "val_acc", "mean_reward"])
    parser.add_argument("--include-rl", action="store_true", help="include RL eval summary from reports/rl_eval.csv if present")
    args = parser.parse_args()

    # load standard eval results
    if not os.path.exists(args.csv):
        print(f"[LAD] aviso: arquivo não encontrado: {args.csv}")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(args.csv)

    # basic cleaning: drop obvious duplicate rows (same run_name+ckpt+timestamp)
    if not df.empty and set(["run_name", "ckpt", "timestamp"]).issubset(df.columns):
        df = df.drop_duplicates(subset=["run_name", "ckpt", "timestamp"])

    # aggregate by run_name + ckpt choosing best row per metric (or compute mean for mean_reward)
    agg = pd.DataFrame()
    if not df.empty:
        if args.metric in ("val_loss", "val_acc") and set(["run_name", "ckpt"]).issubset(df.columns):
            if args.metric == "val_loss":
                # choose row with minimal val_loss per run_name+ckpt
                agg = df.sort_values(by=["run_name", "ckpt", "val_loss"]).groupby(["run_name", "ckpt"], as_index=False).first()
            else:
                # choose row with maximal val_acc per run_name+ckpt
                agg = df.sort_values(by=["run_name", "ckpt", "val_acc"], ascending=[True, True, False]).groupby(["run_name", "ckpt"], as_index=False).first()
        elif args.metric == "mean_reward":
            # compute mean_reward per run_name+ckpt from any existing mean_reward column in df
            if "mean_reward" in df.columns and set(["run_name", "ckpt"]).issubset(df.columns):
                agg = df.groupby(["run_name", "ckpt"], as_index=False)["mean_reward"].mean()
            else:
                agg = pd.DataFrame()

    # optionally include RL evaluation summary
    rl_row = None
    if args.include_rl:
        rl_path = os.path.join("reports", "rl_eval.csv")
        if os.path.exists(rl_path):
            rl_df = pd.read_csv(rl_path)
            if "reward" in rl_df.columns:
                mean_r = float(rl_df["reward"].mean())
                rl_row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_name": "rl_logan_v1",
                    "ckpt": "best",
                    "val_loss": None,
                    "val_acc": None,
                    "mean_reward": mean_r,
                }

                # if we're aggregating mean_reward, merge this value into agg (avoid duplicate RL rows)
                if args.metric == "mean_reward":
                    if agg.empty:
                        agg = pd.DataFrame([{"run_name": rl_row["run_name"], "ckpt": rl_row["ckpt"], "mean_reward": rl_row["mean_reward"]}])
                    else:
                        # insert or replace existing RL row for run_name+ckpt
                        mask = (agg["run_name"] == rl_row["run_name"]) & (agg["ckpt"] == rl_row["ckpt"])
                        if mask.any():
                            agg.loc[mask, "mean_reward"] = rl_row["mean_reward"]
                        else:
                            agg = pd.concat([agg, pd.DataFrame([{"run_name": rl_row["run_name"], "ckpt": rl_row["ckpt"], "mean_reward": rl_row["mean_reward"]}])], ignore_index=True)

    # Prepare output depending on metric
    print("\n[LAD] LEADERBOARD")
    if args.metric in ("val_loss", "val_acc"):
        if agg.empty:
            print("(nenhum resultado de avaliação disponível)")
        else:
            asc = args.metric == "val_loss"
            out_df = agg.sort_values(by=args.metric, ascending=asc)
            print(out_df[["timestamp", "run_name", "ckpt", "val_loss", "val_acc"]].to_string(index=False))
            # if RL row present, show it separately (no val_loss/val_acc)
            if rl_row is not None:
                print("\n[RL] summary:")
                print(f"  run_name: {rl_row['run_name']} mean_reward: {rl_row['mean_reward']:.4f}")
    else:
        # mean_reward view: collect mean_reward values from eval CSV and RL eval file,
        # then group by run_name+ckpt and compute a single mean to avoid duplicates.
        rows = []

        # collect from main eval CSV if it contains a mean_reward column
        if not df.empty and "mean_reward" in df.columns and set(["run_name", "ckpt"]).issubset(df.columns):
            grp = df.groupby(["run_name", "ckpt"], as_index=False)["mean_reward"].mean()
            for _, r in grp.iterrows():
                rows.append({"run_name": r["run_name"], "ckpt": r["ckpt"], "mean_reward": float(r["mean_reward"])})

        # collect from RL eval file (per-episode rewards), compute its mean
        if args.include_rl:
            rl_path = os.path.join("reports", "rl_eval.csv")
            if os.path.exists(rl_path):
                try:
                    rl_df = pd.read_csv(rl_path)
                    if "reward" in rl_df.columns and not rl_df.empty:
                        mean_r = float(rl_df["reward"].mean())
                        rows.append({"run_name": "rl_logan_v1", "ckpt": "best", "mean_reward": mean_r})
                except Exception:
                    pass

        # dedupe by run_name+ckpt keeping the (single) mean
        if not rows:
            print("(nenhum resultado de reward disponível)")
        else:
            out = pd.DataFrame(rows).groupby(["run_name", "ckpt"], as_index=False)["mean_reward"].mean()
            out = out.sort_values(by="mean_reward", ascending=False)
            print(out.to_string(index=False))

    # save aggregated leaderboard
    out_path = "reports/leaderboard.csv"
    try:
        # Build unified leaderboard with columns: timestamp, run_name, ckpt, val_loss, val_acc, mean_reward
        val_info = pd.DataFrame()
        if not df.empty and set(["run_name", "ckpt"]).issubset(df.columns):
            # take the last timestamped val metrics per run_name+ckpt
            if "timestamp" in df.columns:
                val_info = df.sort_values(by="timestamp").groupby(["run_name", "ckpt"], as_index=False).last()[["timestamp", "run_name", "ckpt", "val_loss", "val_acc"]]
            else:
                val_info = df.groupby(["run_name", "ckpt"], as_index=False).first()[["run_name", "ckpt", "val_loss", "val_acc"]]

        # gather mean_reward from eval_results.csv if present
        reward_info = pd.DataFrame()
        if not df.empty and "mean_reward" in df.columns and set(["run_name", "ckpt"]).issubset(df.columns):
            reward_info = df.groupby(["run_name", "ckpt"], as_index=False)["mean_reward"].mean()

        # gather RL per-episode rewards and compute mean
        rl_path = os.path.join("reports", "rl_eval.csv")
        if os.path.exists(rl_path):
            try:
                rl_df = pd.read_csv(rl_path)
                if "reward" in rl_df.columns and not rl_df.empty:
                    mean_r = float(rl_df["reward"].mean())
                    # append/merge into reward_info under run_name rl_logan_v1, ckpt best
                    row = pd.DataFrame([{"run_name": "rl_logan_v1", "ckpt": "best", "mean_reward": mean_r}])
                    if reward_info.empty:
                        reward_info = row
                    else:
                        reward_info = pd.concat([reward_info, row], ignore_index=True).groupby(["run_name", "ckpt"], as_index=False)["mean_reward"].mean()
            except Exception:
                pass

        # merge val_info and reward_info
        if val_info.empty and reward_info.empty:
            # nothing to save, write an empty csv
            pd.DataFrame(columns=["timestamp", "run_name", "ckpt", "val_loss", "val_acc", "mean_reward"]).to_csv(out_path, index=False)
        else:
            if val_info.empty:
                merged = reward_info.copy()
                merged["timestamp"] = datetime.now().isoformat(timespec="seconds")
                merged = merged[["timestamp", "run_name", "ckpt", "mean_reward"]]
                # ensure columns
                merged = merged.reindex(columns=["timestamp", "run_name", "ckpt", "val_loss", "val_acc", "mean_reward"]).fillna(value={"val_loss": pd.NA, "val_acc": pd.NA})
            elif reward_info.empty:
                merged = val_info.copy()
                merged["mean_reward"] = pd.NA
                merged = merged.reindex(columns=["timestamp", "run_name", "ckpt", "val_loss", "val_acc", "mean_reward"])
            else:
                merged = pd.merge(val_info, reward_info, on=["run_name", "ckpt"], how="outer")
                # ensure timestamp exists
                if "timestamp" not in merged.columns:
                    merged["timestamp"] = datetime.now().isoformat(timespec="seconds")
                merged = merged.reindex(columns=["timestamp", "run_name", "ckpt", "val_loss", "val_acc", "mean_reward"]).sort_values(by=["run_name", "ckpt"])

            merged.to_csv(out_path, index=False)
        print(f"\n[LAD] leaderboard salvo em {out_path}")
    except Exception as e:
        print(f"[LAD] falha ao salvar leaderboard: {e}")


if __name__ == "__main__":
    main()
