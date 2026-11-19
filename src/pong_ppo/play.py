"""Запис гри навченого агента у GIF (або показ у вікні з --show).

Запуск:  python -m pong_ppo.play --checkpoint models/pong_ppo_final.pt --gif results/pong_game.gif
"""
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .env import PongEnv
from .model import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/pong_ppo_final.pt")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gif", default="results/pong_game.gif")
    ap.add_argument("--max-frames", type=int, default=500, help="приблизна кількість кадрів у GIF (уся гра)")
    ap.add_argument("--show", action="store_true", help="показати гру у вікні (потрібен matplotlib)")
    args = ap.parse_args()
    net = ActorCritic(); net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)); net.eval()
    def run(stride=None):
        """Грає одну гру; якщо stride задано – записує кожен stride-й крок (гра детермінована для seed)."""
        env = PongEnv(seed=args.seed, sticky=0.25, hit_bonus=0.0)
        obs = env.reset()
        frames, k = [], 0
        font = ImageFont.load_default(size=15)
        if args.show:
            import matplotlib.pyplot as plt
            plt.ion(); im = plt.imshow(env.frame()); plt.axis("off")
        while True:
            a = int(net.greedy(torch.tensor(obs)[None])[0])
            obs, _, term, trunc, info = env.step(a)
            if args.show:
                im.set_data(env.frame()); plt.pause(0.001)
            if stride and (k % stride == 0 or term or trunc):
                img = Image.new("RGB", (320, 446), (20, 20, 20))
                img.paste(Image.fromarray(env.frame()).resize((320, 420), Image.NEAREST), (0, 0))
                ImageDraw.Draw(img).text((8, 425), f"PPO agent {info['score'][1]} : {info['score'][0]} CPU",
                                         fill=(255, 255, 255), font=font)
                frames.append(img)
            k += 1
            if term or trunc:
                return info, k, frames

    info, total, _ = run(None if not args.gif else 10 ** 9)
    if args.gif:
        stride = max(1, total // args.max_frames)
        info, _, frames = run(stride)
    print("Рахунок: агент", info["score"][1], ":", info["score"][0], "суперник")
    if args.gif:
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)
        frames[-1].save(args.gif.replace(".gif", "_last.png"))


if __name__ == "__main__":
    main()
