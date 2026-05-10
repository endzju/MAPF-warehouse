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

    def get_action(self, obs: dict, device="cpu") -> int:

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_actions)

        view = torch.FloatTensor(obs["view"]).unsqueeze(0).to(device)
        goal = torch.FloatTensor(obs["goal_vector"]).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.model(view, goal)

        return torch.argmax(q_values).item()

    def get_actions(self, obs_dict: dict[dict], device="cpu") -> dict[int, int]:
        actions = {}
        network_indices = []
        for idx in obs_dict.keys():
            if np.random.rand() <= self.epsilon:
                actions[idx] = int(np.random.randint(self.n_actions))
            else:
                network_indices.append(idx)

        if not network_indices:
            return actions

        views = [obs_dict[i]["view"] for i in network_indices]
        goals = [obs_dict[i]["goal_vector"] for i in network_indices]

        views_tensor = torch.FloatTensor(np.stack(views)).to(device)
        goals_tensor = torch.FloatTensor(np.stack(goals)).to(device)

        with torch.no_grad():
            q_values = self.model(views_tensor, goals_tensor)
            network_actions = q_values.argmax(dim=1).cpu().numpy()

        for idx, action in zip(network_indices, network_actions):
            actions[idx] = int(action)
        return actions

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
