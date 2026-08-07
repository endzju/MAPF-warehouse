import torch
from torch import nn

from src.neural_networks.architectures.base_model import BaseModel


class CNN(BaseModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cnn_layers = []
        prev_channels = self.view_dims
        for kernel_size, hidden_dims, padding in self.hidden_layers["cnn_layers"]:
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=hidden_dims,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            cnn_layers.append(nn.ReLU())
            prev_channels = hidden_dims
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, self.view_dims, self.view_size, self.view_size)
            prev_size = self.cnn(dummy).shape[1] + self.additional_input_size

        mlp_layers = []
        for hidden_size in self.hidden_layers["mlp_layers"]:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            prev_size = hidden_size

        mlp_layers.append(nn.Linear(prev_size, self.output_size))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, view, additional_input):
        cnn_features = self.cnn(view)
        combined = torch.cat([cnn_features, additional_input], dim=1)
        return self.mlp(combined)

    @property
    def model_name(self):
        cnn_name = "_".join(
            f"k{k}c{c}p{p}" for k, c, p in self.hidden_layers["cnn_layers"]
        )
        mlp_name = "_".join(map(str, self.hidden_layers["mlp_layers"]))
        return f"{self.__class__.__name__}_{cnn_name}_MLP_{mlp_name}"
