from abc import ABC, abstractmethod

from torch import nn

from src.neural_networks.model_config import ModelConfig


class BaseModel(nn.Module, ABC):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.hidden_layers = model_config.hidden_layers
        self.view_size = model_config.view_size
        self.view_dims = model_config.view_dims
        self.additional_input_size = model_config.get_additional_input_size()
        self.output_size = model_config.get_output_size()
        self.view_input_size = model_config.get_view_input_size()

    @property
    @abstractmethod
    def model_name(self):
        pass
