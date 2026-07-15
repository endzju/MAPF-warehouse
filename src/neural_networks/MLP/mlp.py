import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_layers: list[int], input_size: int, output_size: int):
        super(MLP, self).__init__()
        self.display_name = (
            f"{self.__class__.__name__}_{'_'.join(map(str, hidden_layers))}"
        )
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size

        layers = []

        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, view, goal_vec):
        view_flat = view.view(view.size(0), -1)
        combined = torch.cat([view_flat, goal_vec], dim=1)
        return self.network(combined)
