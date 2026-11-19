"""Оцінювання навченої моделі на нових іграх (жадібний вибір дії, ваги не змінюються).

Запуск:  python -m pong_ppo.evaluate --checkpoint models/pong_ppo_final.pt --games 50 --seed 900000
"""
import argparse, json

import numpy as np
import torch

from .env import PongEnv
from .model import ActorCritic


def play_game(net, seed, sticky=0.25):
    env = PongEnv(seed=seed, sticky=sticky, hit_bonus=0.0)
    obs = env.reset()
    while True:
        a = int(net.greedy(torch.tensor(obs)[None])[0])
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/pong_ppo_final.pt")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    net = ActorCritic(); net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)); net.eval()
    games = []
    for k in range(args.games):
        info = play_game(net, args.seed + k, args.sticky)
        games.append({"seed": args.seed + k, "cpu": info["score"][0], "agent": info["score"][1],
                      "won": info["won"], "steps": info["steps"], "hits": info["hits"]})
        print(f"гра {k + 1:3d}: агент {info['score'][1]:2d} : {info['score'][0]:2d} суперник")
    wins = sum(g["won"] for g in games)
    diff = np.array([g["agent"] - g["cpu"] for g in games])
    summary = {"checkpoint": args.checkpoint, "games": args.games, "seed_from": args.seed, "sticky_actions": args.sticky,
               "wins": wins, "win_rate": wins / args.games, "score_diff_mean": float(diff.mean()),
               "agent_points_mean": float(np.mean([g["agent"] for g in games])),
               "cpu_points_mean": float(np.mean([g["cpu"] for g in games])),
               "shutouts_21_0": sum(1 for g in games if g["agent"] == 21 and g["cpu"] == 0), "per_game": games}
    print(f"\nПеремог: {wins}/{args.games} ({100 * wins / args.games:.0f}%), середня різниця рахунку {diff.mean():+.2f}")
    if args.out:
        json.dump(summary, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
