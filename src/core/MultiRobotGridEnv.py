import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiRobotGridEnv(gym.Env):
    def __init__(
        self,
        grid_size: tuple[int, int] = (10, 10),
        num_agents: int = 5,
        obstacles: list[tuple[int, int]] | None = None,
    ):
        super(MultiRobotGridEnv, self).__init__()
        self.grid_width, self.grid_height = grid_size
        self.num_states = self.grid_width * self.grid_height
        self.num_agents = num_agents
        self.obstacles = np.zeros((self.grid_width, self.grid_height), dtype=np.int32)

        if obstacles:
            obs = np.array(obstacles, dtype=np.int32)
            self.obstacles[obs[:, 0], obs[:, 1]] = 1

        # Akcje: 0=Góra, 1=Dół, 2=Lewo, 3=Prawo, 4=Czekaj
        # MultiDiscrete pozwala zdefiniować akcję dla każdego robota naraz
        self.action_space = spaces.MultiDiscrete([5] * num_agents)

        self.observation_space = spaces.Box(
            low=0,
            high=num_agents,
            shape=(3, self.grid_width, self.grid_height),
            dtype=np.int32,
        )

    def _non_obstacle_cells(self):
        return [
            (x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if self.obstacles[x, y] == 0
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        robot_indices = self.np_random.integers(
            low=0, high=self.num_states, size=self.num_agents
        )
        goal_indices = self.np_random.integers(
            low=0, high=self.num_states, size=self.num_agents
        )

        return self._get_obs(), {}

    def step(self, actions):
        # 1. Zastosuj ruchy (pamiętaj o ścianach!)
        # 2. Sprawdź kolizje między robotami
        # 3. Oblicz nagrodę (Reward)
        # 4. Sprawdź czy wszyscy dotarli do celu (Terminated)

        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._is_done()
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.agent_pos.flatten(),
                self.goals.flatten(),
            ]
        )
