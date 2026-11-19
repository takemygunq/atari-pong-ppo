"""Actor-Critic з окремими вагами (як у SnakeAI v9), але повнозв'язний: стан – вектор із 10 чисел."""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import OBS_DIM, ACTIONS


def mlp(out_dim: int, hidden=(256, 128)) -> nn.Sequential:
    layers, d = [], OBS_DIM
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, hidden=(256, 128)):
        super().__init__()
        self.actor = mlp(len(ACTIONS), hidden)     # 3 логіти: стоп / вгору / вниз
        self.critic = mlp(1, hidden)               # оцінка майбутньої винагороди
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        dist = Categorical(logits=self.actor(x))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(x)

    @torch.no_grad()
    def greedy(self, x):
        return self.actor(x).argmax(-1)
