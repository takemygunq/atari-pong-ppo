"""Навчання агента Pong методом PPO (Proximal Policy Optimization) на CPU.

Запуск:  python -m pong_ppo.train --config configs/training.json
"""
import argparse, csv, json, os, time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .env import VecPong
from .model import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.json")
    ap.add_argument("--out", default="runs/latest")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.set_num_threads(1)

    envs = VecPong(cfg["num_envs"], seed=cfg["seed"], hit_bonus=cfg["rewards"]["hit_bonus"],
                   sticky=cfg["sticky_actions"])
    net = ActorCritic(tuple(cfg["hidden"]))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    N, T = cfg["num_envs"], cfg["horizon"]
    batch = N * T
    iters = cfg["total_steps"] // batch
    obs_b = torch.zeros((T, N, 10)); act_b = torch.zeros((T, N), dtype=torch.long)
    logp_b = torch.zeros((T, N)); rew_b = torch.zeros((T, N)); done_b = torch.zeros((T, N)); val_b = torch.zeros((T, N))

    obs = torch.tensor(envs.reset())
    done = torch.zeros(N)
    recent = deque(maxlen=100)       # (різниця рахунку, перемога, відбиття)
    log = open(os.path.join(args.out, "training_log.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["iteration", "env_steps", "games", "score_diff_mean", "win_rate", "hits_mean",
                 "policy_loss", "value_loss", "entropy", "seconds"])
    games, t0, best = 0, time.time(), -99.0

    for it in range(1, iters + 1):
        frac = 1.0 - (it - 1) / iters
        opt.param_groups[0]["lr"] = cfg["learning_rate"] * frac          # лінійне зменшення lr
        # ---------------------------------------------------- збір досвіду
        for t in range(T):
            obs_b[t] = obs; done_b[t] = done
            with torch.no_grad():
                a, lp, _, v = net.act(obs)
            act_b[t] = a; logp_b[t] = lp; val_b[t] = v
            o, r, d, infos = envs.step(a.numpy())
            rew_b[t] = torch.tensor(r)
            obs = torch.tensor(o); done = torch.tensor(d)
            for info in infos:
                if info.get("episode_end"):
                    games += 1
                    s = info["score"]
                    recent.append((s[1] - s[0], float(info["won"]), info["hits"]))
        # ---------------------------------------------------- GAE
        with torch.no_grad():
            next_v = net.value(obs)
        adv = torch.zeros_like(rew_b); last = torch.zeros(N)
        for t in reversed(range(T)):
            nonterm = 1.0 - (done if t == T - 1 else done_b[t + 1])
            nv = next_v if t == T - 1 else val_b[t + 1]
            delta = rew_b[t] + cfg["gamma"] * nv * nonterm - val_b[t]
            last = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterm * last
            adv[t] = last
        ret = adv + val_b
        # ---------------------------------------------------- оновлення PPO
        b_obs, b_act, b_logp = obs_b.reshape(-1, 10), act_b.reshape(-1), logp_b.reshape(-1)
        b_adv, b_ret, b_val = adv.reshape(-1), ret.reshape(-1), val_b.reshape(-1)
        mb = cfg["minibatch_size"]
        for _ in range(cfg["epochs"]):
            idx = torch.randperm(batch)
            for s in range(0, batch, mb):
                j = idx[s:s + mb]
                _, lp, ent, v = net.act(b_obs[j], b_act[j])
                ratio = (lp - b_logp[j]).exp()
                a_n = (b_adv[j] - b_adv[j].mean()) / (b_adv[j].std() + 1e-8)
                pg = torch.max(-a_n * ratio, -a_n * ratio.clamp(1 - cfg["ppo_clip"], 1 + cfg["ppo_clip"])).mean()
                v_clip = b_val[j] + (v - b_val[j]).clamp(-cfg["value_clip"], cfg["value_clip"])
                vl = 0.5 * torch.max((v - b_ret[j]) ** 2, (v_clip - b_ret[j]) ** 2).mean()
                loss = pg + cfg["value_coef"] * vl - cfg["entropy_coefficient"] * ent.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip"]); opt.step()
        # ---------------------------------------------------- журнал
        if recent:
            sd = float(np.mean([x[0] for x in recent])); wrate = float(np.mean([x[1] for x in recent]))
            hm = float(np.mean([x[2] for x in recent]))
        else:
            sd = wrate = hm = float("nan")
        wr.writerow([it, it * batch, games, f"{sd:.3f}", f"{wrate:.3f}", f"{hm:.2f}", f"{pg.item():.4f}",
                     f"{vl.item():.4f}", f"{ent.mean().item():.4f}", f"{time.time() - t0:.0f}"])
        log.flush()
        if it % cfg["save_every"] == 0 or it == iters:
            torch.save(net.state_dict(), os.path.join(args.out, f"iter_{it:05d}.pt"))
            if recent and len(recent) >= 20 and sd > best:
                best = sd
                torch.save(net.state_dict(), os.path.join(args.out, "best.pt"))
            print(f"it {it}/{iters} steps {it * batch} games {games} diff {sd:+.2f} win {wrate:.2f} "
                  f"hits {hm:.1f} ent {ent.mean().item():.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), os.path.join(args.out, "final.pt"))


if __name__ == "__main__":
    main()
