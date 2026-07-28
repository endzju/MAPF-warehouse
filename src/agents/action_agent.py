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

        indices = np.fromiter(obs_dict.keys(), dtype=np.int32)
        mask = np.random.rand(len(indices)) > self.epsilon
        network_indices = indices[mask]
        random_indices = indices[~mask]
        random_actions = np.random.randint(self.n_actions, size=len(random_indices))
        for idx, action in zip(random_indices, random_actions, strict=True):
            actions[idx] = int(action)

        if network_indices.size == 0:
            return actions

        views_tensor = torch.as_tensor(
            np.array([obs_dict[i]["view"] for i in network_indices], dtype=np.float32),
            device=device,
        )
        goals_tensor = torch.as_tensor(
            np.array(
                [obs_dict[i]["goal_vector"] for i in network_indices], dtype=np.float32
            ),
            device=device,
        )

        with torch.inference_mode():
            q_values = self.model(views_tensor, goals_tensor)
            network_actions = q_values.argmax(dim=1).cpu()

        for idx, action in zip(network_indices, network_actions, strict=True):
            actions[idx] = int(action)

        return actions

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
