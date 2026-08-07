from abc import ABC, abstractmethod

from torch import nn


class BaseModel(nn.Module, ABC):
    def __init__(
        self,
        hidden_layers,
        view_size,
        view_dims,
        additional_input_size,
        output_size,
        observation_config,
        observation_config_name="default",
        save_path=None,
        checkpoint_name=None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.view_size = view_size
        self.view_dims = view_dims
        self.additional_input_size = additional_input_size
        self.output_size = output_size
        self.observation_config = observation_config
        self.observation_config_name = observation_config_name
        self.view_input_size = view_dims * view_size * view_size
        self.save_path = save_path
        self.checkpoint_name = checkpoint_name

    def get_model_params(self) -> dict:
        return {
            "hidden_layers": self.hidden_layers,
            "view_size": self.view_size,
            "view_dims": self.view_dims,
            "additional_input_size": self.additional_input_size,
            "output_size": self.output_size,
            "observation_config": self.observation_config,
            "observation_config_name": self.observation_config_name,
            "save_path": self.save_path,
            "checkpoint_name": self.checkpoint_name,
        }

    @property
    @abstractmethod
    def model_name(self):
        pass
