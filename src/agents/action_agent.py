import numpy as np
import torch
from torch import nn


class ActionAgent:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        decay: float = 0.995,
    ):
        self.model = model
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.n_actions = 5

    def get_action(self, obs: dict, device="cpu"):

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_actions)

        view = torch.FloatTensor(obs["view"]).unsqueeze(0).to(device)
        goal = torch.FloatTensor(obs["goal_vector"]).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.model(view, goal)

        return torch.argmax(q_values).item()

    def get_actions(self, obs_list: list[dict], device="cpu"):

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_actions)

        views = [o["view"] for o in obs_list]
        goals = [o["goal_vector"] for o in obs_list]

        views_tensor = torch.FloatTensor(np.stack(views)).to(device)
        goals_tensor = torch.FloatTensor(np.stack(goals)).to(device)

        with torch.no_grad():
            q_values = self.model(views_tensor, goals_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()

        return actions

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
