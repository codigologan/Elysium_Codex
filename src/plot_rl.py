import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "reports/rl_results.csv"
if not os.path.exists(csv_path):
    print(f"arquivo não encontrado: {csv_path}")
else:
    df = pd.read_csv(csv_path)
    if "mean_reward_last_50" in df.columns:
        plt.figure(figsize=(8,4))
        plt.plot(df["mean_reward_last_50"].values, marker="o")
        plt.xlabel("Run index")
        plt.ylabel("Mean reward (last 50)")
        plt.title("Logan RL Leaderboard Metric")
        plt.tight_layout()
        plt.show()
    else:
        print("coluna 'mean_reward_last_50' não encontrada no CSV")
