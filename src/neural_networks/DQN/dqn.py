import torch
from torch import nn


class DQNet1(nn.Module):
    def __init__(self, view_shape, goal_vec_size, n_actions):
        super(DQNet1, self).__init__()

        view_flat_size = view_shape[0] * view_shape[1] * view_shape[2]

        self.network = nn.Sequential(
            nn.Linear(view_flat_size + goal_vec_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, view, goal_vec):
        view_flat = view.view(view.size(0), -1)

        combined = torch.cat([view_flat, goal_vec], dim=1)

        return self.network(combined)


class DQNet2(nn.Module):
    def __init__(self, view_shape, goal_vec_size, n_actions):
        super(DQNet2, self).__init__()

        view_flat_size = view_shape[0] * view_shape[1] * view_shape[2]

        self.network = nn.Sequential(
            nn.Linear(view_flat_size + goal_vec_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, view, goal_vec):
        view_flat = view.view(view.size(0), -1)

        combined = torch.cat([view_flat, goal_vec], dim=1)

        return self.network(combined)


class DQNet3(nn.Module):
    def __init__(self, view_shape, goal_vec_size, n_actions):
        super(DQNet3, self).__init__()

        view_flat_size = view_shape[0] * view_shape[1] * view_shape[2]

        self.network = nn.Sequential(
            nn.Linear(view_flat_size + goal_vec_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, view, goal_vec):
        view_flat = view.view(view.size(0), -1)

        combined = torch.cat([view_flat, goal_vec], dim=1)

        return self.network(combined)
