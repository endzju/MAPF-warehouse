import torch
import torch.nn as nn


class CNN1(nn.Module):
    def __init__(self, view_shape, goal_vec_size, n_actions):
        super(CNN1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(view_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out_size = 64 * view_shape[1] * view_shape[2]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size + goal_vec_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, view, goal_vec):
        cnn_features = self.cnn(view)
        combined = torch.cat([cnn_features, goal_vec], dim=1)
        return self.fc(combined)
