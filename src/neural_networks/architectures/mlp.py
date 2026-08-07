import torch
from torch import nn

from src.neural_networks.architectures.base_model import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        layers = []
        prev_size = self.view_input_size + self.additional_input_size
        for hidden_size in self.hidden_layers["mlp_layers"]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, view, additional_input):
        view_flat = view.view(view.size(0), -1)
        combined = torch.cat([view_flat, additional_input], dim=1)
        return self.network(combined)

    @property
    def model_name(self):
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.hidden_layers['mlp_layers']))}"
