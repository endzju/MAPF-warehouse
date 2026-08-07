from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationConfig:
    float_goal_vector: bool = False
    modulo_reward_x: tuple[int] = ()
    modulo_reward_y: tuple[int] = ()
    modulo_depot_distance_reward_cancel: int = 5
    view_dims: int = 4
    goal_vec_size: int = 2
    n_actions: int = 5

    def get_additional_input_size(self) -> int:
        return (
            self.goal_vec_size + len(self.modulo_reward_x) + len(self.modulo_reward_y)
        )

    def get_output_size(self) -> int:
        return self.n_actions

    def to_dict(self) -> dict:
        return {
            "float_goal_vector": self.float_goal_vector,
            "modulo_reward_x": self.modulo_reward_x,
            "modulo_reward_y": self.modulo_reward_y,
            "modulo_depot_distance_reward_cancel": self.modulo_depot_distance_reward_cancel,
            "view_dims": self.view_dims,
            "goal_vec_size": self.goal_vec_size,
            "n_actions": self.n_actions,
        }


DEFAULT = ObservationConfig(
    float_goal_vector=False,
)

FLOAT_GOAL_VECTOR = ObservationConfig(
    float_goal_vector=True,
)

MODULO2 = ObservationConfig(
    float_goal_vector=False,
    modulo_reward_x=(0, 0),
    modulo_reward_y=(0, 0),
)

MODULO3 = ObservationConfig(
    float_goal_vector=False,
    modulo_reward_x=(0, 0, 0),
    modulo_reward_y=(0, 0, 0),
)

MODULO4 = ObservationConfig(
    float_goal_vector=False,
    modulo_reward_x=(0, 0, 0, 0),
    modulo_reward_y=(0, 0, 0, 0),
)

OBSERVATION_CONFIGS = {
    "default": DEFAULT,
    "float_goal_vector": FLOAT_GOAL_VECTOR,
    "modulo2": MODULO2,
    "modulo3": MODULO3,
    "modulo4": MODULO4,
}
