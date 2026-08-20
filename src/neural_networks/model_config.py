from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModelConfig:
    model_class: type
    hidden_layers: dict
    view_size: int = 7
    view_dims: int = 4
    goal_vec_size: int = 2
    n_actions: int = 5
    float_goal_vector: bool = True
    modulo_reward_x: tuple[int] = ()
    modulo_reward_y: tuple[int] = ()
    modulo_goal_distance_reward_cancel: int = 5
    x_position_float: bool = False
    y_position_float: bool = False
    num_robots: int = 60
    num_tasks: int = 5 * num_robots
    task_tsp: bool = False
    suffix: str = ""

    batch_size: int = 4096 * 4
    num_batches: int = 100
    buffer_length: int = 1024**2
    target_update_interval: int = 60
    gamma: float = 0.99

    def build_model(self):
        """Create model instance."""
        return self.model_class(model_config=self)

    def load_model(self):
        """Create model instance and load weights from file."""
        model = self.build_model()
        model.load_state_dict(torch.load(model.get_model_path(), weights_only=True))
        return model

    def get_view_input_size(self) -> int:
        return self.view_dims * self.view_size * self.view_size

    def get_view_input_shape(self) -> tuple[int, int, int]:
        return self.view_dims, self.view_size, self.view_size

    def get_additional_input_size(self) -> int:
        return (
            self.goal_vec_size
            + len(self.modulo_reward_x)
            + len(self.modulo_reward_y)
            + int(self.x_position_float)
            + int(self.y_position_float)
        )

    def get_output_size(self) -> int:
        return self.n_actions

    def get_model_dir_name(self):
        name = ""
        if "cnn_layers" in self.hidden_layers:
            name += "CNN_" + "_".join(
                f"k{k}c{c}p{p}" for k, c, p in self.hidden_layers["cnn_layers"]
            )
        if "mlp_layers" in self.hidden_layers:
            name += "MLP_" + "_".join(map(str, self.hidden_layers["mlp_layers"]))

        return name

    def get_model_dir_path(self):
        return Path(__file__).parent / "models" / self.get_model_dir_name()

    def get_model_path(self):
        return self.get_model_dir_path() / f"{self.get_params_string()}.pth"

    def get_model_full_name(self):
        return f"{self.get_model_dir_name()}_{self.get_params_string()}"

    def get_params_string(self) -> str:
        """Generuje czytelną nazwę na podstawie parametrów konfiguracyjnych."""
        parts = []

        parts.append(f"v{self.view_size}")
        parts.append(f"r{self.num_robots}")
        parts.append(f"u{self.target_update_interval}")
        parts.append(f"l{self.buffer_length}")
        parts.append(f"s{self.batch_size}")
        parts.append(f"b{self.num_batches}")
        parts.append(f"g{self.gamma:.3f}".replace(".", ""))

        obs_parts = []
        if not getattr(self, "float_goal_vector", True):
            obs_parts.append("int_goal")
        if self.modulo_reward_x:
            obs_parts.append(f"modX{self.modulo_reward_x}")
        if self.modulo_reward_y:
            obs_parts.append(f"modY{self.modulo_reward_y}")
        if self.modulo_reward_x or self.modulo_reward_y:
            obs_parts.append(f"dis{self.modulo_goal_distance_reward_cancel}")

        if self.x_position_float and self.y_position_float:
            obs_parts.append("floatposXY")
        elif self.x_position_float:
            obs_parts.append("floatposX")
        elif self.y_position_float:
            obs_parts.append("floatposY")

        if self.task_tsp:
            obs_parts.append("tsp")

        if self.suffix:
            obs_parts.append(self.suffix)

        # Jeśli nie dodano żadnych specyficznych modyfikatorów, oznaczamy jako "default"
        obs_name = "_".join(obs_parts) if obs_parts else "default"
        parts.append(obs_name)

        return "_".join(parts)

    def get_env_params(self) -> dict:
        return {
            "agent_view_size": self.view_size,
            "num_tasks": self.num_tasks,
            "max_robots": self.num_robots,
            "modulo_reward_x": self.modulo_reward_x,
            "modulo_reward_y": self.modulo_reward_y,
            "modulo_goal_distance_reward_cancel": self.modulo_goal_distance_reward_cancel,
            "float_goal_vector": self.float_goal_vector,
            "view_dims": self.view_dims,
            "goal_vec_size": self.goal_vec_size,
            "n_actions": self.n_actions,
            "x_position_float": self.x_position_float,
            "y_position_float": self.y_position_float,
            "task_tsp": self.task_tsp,
        }

    def get_train_params(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "buffer_length": self.buffer_length,
            "target_update_interval": self.target_update_interval,
            "gamma": self.gamma,
        }
