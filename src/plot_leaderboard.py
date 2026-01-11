import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("reports/leaderboard.csv")

    df = df.sort_values("val_loss")

    plt.figure(figsize=(8,4))
    plt.bar(df["run_name"], df["val_loss"])
    plt.ylabel("Validation Loss")
    plt.title("LAD Leaderboard (lower is better)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
