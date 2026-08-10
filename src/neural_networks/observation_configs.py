from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationConfig:
    float_goal_vector: bool = True
    modulo_reward_x: tuple[int] = ()
    modulo_reward_y: tuple[int] = ()
    modulo_goal_distance_reward_cancel: int = 5
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
            "modulo_goal_distance_reward_cancel": self.modulo_goal_distance_reward_cancel,
            "view_dims": self.view_dims,
            "goal_vec_size": self.goal_vec_size,
            "n_actions": self.n_actions,
        }


DEFAULT = ObservationConfig()

INTGOALVECTOR = ObservationConfig(
    float_goal_vector=False,
)

MODULO2 = ObservationConfig(
    modulo_reward_x=(0, 0),
    modulo_reward_y=(0, 0),
)

MODULO2REWARDX = ObservationConfig(
    modulo_reward_x=(1, -1),
    modulo_reward_y=(0, 0),
)
MODULO2REWARDY = ObservationConfig(
    modulo_reward_x=(0, 0),
    modulo_reward_y=(1, -1),
)
MODULO2REWARDXY = ObservationConfig(
    modulo_reward_x=(1, -1),
    modulo_reward_y=(1, -1),
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

MODULO4SINGLELANE1 = ObservationConfig(
    float_goal_vector=False,
    modulo_reward_x=(0, 1, -1, 0),
    modulo_reward_y=(0, 1, -1, 0),
)

MODULO4SINGLELANE2 = ObservationConfig(
    float_goal_vector=False,
    modulo_reward_x=(0, 1, 0, -1),
    modulo_reward_y=(0, 1, 0, -1),
)

MODULO4DOUBLELANE = ObservationConfig(
    float_goal_vector=False,
    modulo_reward_x=(1, 1, -1, -1),
    modulo_reward_y=(1, 1, -1, -1),
)

OBSERVATION_CONFIGS = {
    "default": DEFAULT,
    "modulo2": MODULO2,
    "modulo3": MODULO3,
    "modulo4": MODULO4,
    "modulo2rewardx": MODULO2REWARDX,
    "modulo2rewardy": MODULO2REWARDY,
    "modulo2rewardxy": MODULO2REWARDXY,
    "intgoalvector": INTGOALVECTOR,
    "modulo4singlelane1": MODULO4SINGLELANE1,
    "modulo4singlelane2": MODULO4SINGLELANE2,
    "modulo4doublelane": MODULO4DOUBLELANE,
}
