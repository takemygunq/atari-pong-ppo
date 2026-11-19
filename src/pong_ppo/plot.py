"""Графіки історії навчання з training_log.csv.

Запуск:  python -m pong_ppo.plot --log results/training_log.csv --out results/training_history.png
"""
import argparse, csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/training_log.csv")
    ap.add_argument("--out", default="results/training_history.png")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.log)) if r["score_diff_mean"] != "nan"]
    x = [int(r["env_steps"]) / 1e6 for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(x, [float(r["score_diff_mean"]) for r in rows], color="#3b6fb6")
    ax[0].axhline(0, color="#999", lw=0.8, ls="--")
    ax[0].set_title("Різниця рахунку (агент − суперник)"); ax[0].set_ylim(-21.5, 21.5)
    ax[1].plot(x, [100 * float(r["win_rate"]) for r in rows], color="#2e7d32")
    ax[1].set_title("Частка перемог, %"); ax[1].set_ylim(-3, 103)
    ax[2].plot(x, [float(r["hits_mean"]) for r in rows], color="#e08a2c")
    ax[2].set_title("Відбиттів м'яча за гру")
    for a in ax:
        a.set_xlabel("Кроки середовища, млн"); a.grid(color="#e5e5e5")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.text(0.5, -0.04, "Середнє за останні 100 навчальних ігор (дії вибираються випадково з імовірностей мережі)",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
