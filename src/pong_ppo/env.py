"""Середовище Atari Pong з компактним станом, зчитаним з пам'яті приставки (RAM).

Як і в SnakeAI, мережа отримує не сирі пікселі, а підготовлений стан гри.
Координати м'яча й ракеток беруться з RAM Atari 2600 (ALE):
    ram[49] – м'яч X, ram[54] – м'яч Y (0 – м'яча немає на полі),
    ram[51] – ракетка гравця (праворуч), ram[50] – ракетка суперника (ліворуч),
    ram[13] / ram[14] – рахунок суперника / гравця.
"""
from __future__ import annotations

import numpy as np
from ale_py.env import AtariEnv

ACTIONS = (0, 2, 3)              # NOOP, ВГОРУ (RIGHT), ВНИЗ (LEFT) – мінімальний набір дій
ACTION_NAMES = ("стоп", "вгору", "вниз")
OBS_DIM = 10
PLAYER_X = 188                   # приблизна X-координата ракетки гравця


class PongEnv:
    """Одна гра Pong (до 21 очка) зі станом із 10 чисел і формуванням винагороди."""

    def __init__(self, seed: int = 0, sticky: float = 0.25, hit_bonus: float = 0.1,
                 max_steps: int = 27_000, render: bool = False):
        self.ale_env = AtariEnv(game="pong", obs_type="rgb" if render else "ram", frameskip=4,
                                repeat_action_probability=sticky, full_action_space=False)
        self.ale = self.ale_env.ale
        self.seed = seed
        self.hit_bonus = hit_bonus
        self.max_steps = max_steps
        self._episode = 0

    # -------------------------------------------------------------- стан
    def _read(self):
        ram = self.ale.getRAM()
        return (int(ram[49]), int(ram[54]), int(ram[51]), int(ram[50]), int(ram[13]), int(ram[14]))

    def _features(self):
        bx, by, py, cy, _, _ = self.cur
        pbx, pby = self.prev[0], self.prev[1]
        active = 1.0 if by != 0 and pby != 0 else 0.0
        vx = (bx - pbx) * active
        vy = (by - pby) * active
        return np.array([
            (py - 120) / 85.0,                   # ракетка гравця
            (cy - 120) / 85.0,                   # ракетка суперника
            (bx - 120) / 85.0 * active,          # м'яч X
            (by - 120) / 85.0 * active,          # м'яч Y
            vx / 12.0,                           # швидкість м'яча X
            vy / 12.0,                           # швидкість м'яча Y
            (by - py - 8) / 85.0 * active,       # м'яч відносно центру ракетки
            (PLAYER_X - bx) / 170.0 * active,    # відстань м'яча до ракетки по X
            active,                              # м'яч у грі
            1.0 if vx > 0 else 0.0,              # м'яч летить до гравця
        ], dtype=np.float32)

    # -------------------------------------------------------------- API
    def reset(self):
        self.ale_env.reset(seed=self.seed * 100_003 + self._episode)
        self._episode += 1
        self.cur = self.prev = self._read()
        self.steps = 0
        self.hits = 0
        self.score = [0, 0]                  # суперник, гравець
        return self._features()

    def step(self, action: int):
        _, reward, terminated, truncated, _ = self.ale_env.step(ACTIONS[action])
        self.prev, self.cur = self.cur, self._read()
        self.steps += 1
        shaped = float(reward)
        # відбиття м'яча гравцем: м'яч летів праворуч, біля ракетки змінив напрям
        vx_prev = self.prev[0] - self.prev_x2 if hasattr(self, "prev_x2") else 0
        vx_now = self.cur[0] - self.prev[0]
        if self.cur[1] != 0 and vx_prev > 0 and vx_now < 0 and self.prev[0] > 150:
            self.hits += 1
            shaped += self.hit_bonus
        self.prev_x2 = self.prev[0]
        if reward > 0:
            self.score[1] += 1
        elif reward < 0:
            self.score[0] += 1
        truncated = truncated or self.steps >= self.max_steps
        info = {"score": tuple(self.score), "hits": self.hits, "steps": self.steps,
                "won": self.score[1] > self.score[0]}
        return self._features(), shaped, bool(terminated), bool(truncated), info

    def frame(self):
        return self.ale.getScreenRGB()


class VecPong:
    """Простий синхронний вектор середовищ з автоматичним перезапуском ігор."""

    def __init__(self, n: int, seed: int = 0, **kw):
        self.envs = [PongEnv(seed=seed + i, **kw) for i in range(n)]
        self.n = n

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rew, done, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(int(a))
            if term or trunc:
                info["episode_end"] = True
                o = e.reset()
            obs.append(o); rew.append(r); done.append(term or trunc); infos.append(info)
        return np.stack(obs), np.array(rew, np.float32), np.array(done, np.float32), infos
