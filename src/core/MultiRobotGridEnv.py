import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.agents.delivery_robot import DeliveryRobot
from src.utils.enums import Task


class MultiRobotGridEnv(gym.Env):
    def __init__(
        self,
        grid_size: tuple[int, int] = (10, 10),
        num_agents: int = 5,
        agent_view_size: int = 5,
        obstacles: list[tuple[int, int]] | None = None,
        depot: tuple[int, int] = (0, 0),
    ):
        super(MultiRobotGridEnv, self).__init__()
        self.grid_width, self.grid_height = grid_size
        self.num_states = self.grid_width * self.grid_height
        self.num_agents = num_agents
        self.obstacles = np.zeros((self.grid_width, self.grid_height), dtype=np.uint8)
        self.agent_view_size = agent_view_size
        self.agents: set[DeliveryRobot] = set()

        if obstacles:
            obs = np.array(obstacles, dtype=np.uint8)
            self.obstacles[obs[:, 0], obs[:, 1]] = 1

        self.depot = depot

        # Akcje: 0=Góra, 1=Dół, 2=Lewo, 3=Prawo, 4=Czekaj
        # MultiDiscrete pozwala zdefiniować akcję dla każdego robota naraz
        self.action_space = spaces.MultiDiscrete([5] * num_agents)

        self.observation_space = spaces.Dict(
            {
                # obstacles, other agent pos, other agnet goal pos, agent goal pos
                "view": spaces.Box(
                    low=0,
                    high=1,
                    shape=(4, self.agent_view_size, self.agent_view_size),
                    dtype=np.uint8,
                ),
                "goal_vector": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

    def _non_obstacle_cells(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if self.obstacles[x, y] == 0 and (x, y) != self.depot
        ]
        
    def get_empty_cells(self) -> set[tuple[int, int]]:
        occupied_cells = set()
        for agent in self.agents:
            occupied_cells.update(agent.get_occupied_cells())
        indices = np.argwhere(self.obstacles == 1)
        occupied_cells.update(map(tuple, indices))
        occupied_cells.add(self.depot)
        all_cells = {(x, y) for x in range(self.grid_width) for y in range(self.grid_height)}
        empty_cells = all_cells - occupied_cells
        return empty_cells

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        empty_cells = self._non_obstacle_cells()
        self.agents.clear()
        agent_indices = self.np_random.choice(
            len(empty_cells), size=self.num_agents, replace=False
        )
        goal_indices = self.np_random.choice(
            len(empty_cells), size=self.num_agents, replace=True
        )
        observations = {}

        for i, (agent_i, goal_i) in enumerate(zip(agent_indices, goal_indices)):
            agent_pos = empty_cells[agent_i]
            goal_pos = empty_cells[goal_i]
            robot = DeliveryRobot(position=agent_pos, goal=goal_pos, id=i)
            self.agents.add(robot)
            observations[f"agent_{i}"] = self._get_obs(robot)

        return observations, {}

    def step(self, actions: dict[int, int]):
        # actions to array np. [akcja_robota_0, akcja_robota_1, ...]
        observations = {}
        rewards = {}
        terminateds = {"__all__": False}
        
        empty_cells = self.get_empty_cells()
        
        agent_list = list(self.agents)
        
        for agent in self.np_random.shuffle(agent_list):
            # TODO DOKOŃCZYĆ TUTAJ

        for i, agent in enumerate(sorted(list(self.agents), key=lambda x: x.id)):
            # 1. Logika ruchu agenta na podstawie actions[i]
            # 2. Sprawdzenie kolizji
            # 3. Aktualizacja pozycji agenta
            pass

        # Po wykonaniu wszystkich ruchów:
        for i, agent in enumerate(sorted(list(self.agents), key=lambda x: x.id)):
            observations[f"agent_{i}"] = self._get_obs(agent)
            rewards[f"agent_{i}"] = self._calculate_reward(agent)

        # Logika końca epizodu
        terminated = all(a.task_finished for a in self.agents)

        return observations, sum(rewards.values()), terminated, False, {}

    def _in_view(self, agent: DeliveryRobot, pos: tuple[int, int]) -> bool:
        if pos is None:
            return False
        radius = self.agent_view_size // 2
        ax, ay = agent.pos
        return (
            ax - radius <= pos[0] <= ax + radius
            and ay - radius <= pos[1] <= ay + radius
        )

    def _view_position(
        self, agent: DeliveryRobot, pos: tuple[int, int]
    ) -> tuple[int, int]:
        radius = self.agent_view_size // 2
        ax, ay = agent.pos
        return (pos[0] - ax + radius, pos[1] - ay + radius)

    def _add_if_in_view(
        self, agent: DeliveryRobot, pos: tuple[int, int], view: np.ndarray
    ):
        if self._in_view(agent, pos):
            view_pos = self._view_position(agent, pos)
            view[view_pos] = 1

    def _goal_vector(self, agent: DeliveryRobot) -> np.ndarray:
        goal_pos = None
        match agent.task:
            case Task.PICKUP:
                goal_pos = agent.goal
            case Task.LEAVE:
                goal_pos = self.depot

        if goal_pos is None:
            raise ValueError("Agent has no goal")
        max_dist = max(self.grid_width, self.grid_height)
        goal_vec = (
            (goal_pos[0] - agent.pos[0]) / max_dist,
            (goal_pos[1] - agent.pos[1]) / max_dist,
        )
        return np.array(goal_vec, dtype=np.float32)

    def _get_obs(self, agent: DeliveryRobot) -> dict[str, np.ndarray]:
        radius = self.agent_view_size // 2
        padded_grid = np.pad(
            self.obstacles, pad_width=radius, mode="constant", constant_values=1
        )
        ax, ay = agent.pos
        padded_ax = ax + radius
        padded_ay = ay + radius
        obstacles = padded_grid[
            padded_ax - radius : padded_ax + radius + 1,
            padded_ay - radius : padded_ay + radius + 1,
        ]
        other_agent_positions = np.zeros(
            shape=(self.agent_view_size, self.agent_view_size), dtype=np.uint8
        )
        other_agent_goals = np.zeros(
            shape=(self.agent_view_size, self.agent_view_size), dtype=np.uint8
        )
        agent_goal = np.zeros(
            shape=(self.agent_view_size, self.agent_view_size), dtype=np.uint8
        )

        for other_agent in self.agents:
            if other_agent == agent:
                continue
            for cell in other_agent.get_occupied_cells():
                self._add_if_in_view(agent, cell, other_agent_positions)

            self._add_if_in_view(agent, other_agent.goal, other_agent_goals)

        self._add_if_in_view(agent, agent.goal, agent_goal)

        goal_vector = self._goal_vector(agent)

        return {
            "view": np.stack(
                [obstacles, other_agent_positions, other_agent_goals, agent_goal]
            ),
            "goal_vector": goal_vector,
        }

    def render(self, mode="human") -> str:
        grid = np.full((self.grid_width, self.grid_height), ".", dtype=str)
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.obstacles[x, y]:
                    grid[x, y] = "#"
        gx, gy = self.depot
        grid[gx, gy] = "D"
        for agent in self.agents:
            if agent.task == Task.PICKUP:
                ax, ay = agent.goal
                grid[ax, ay] = "o"
        for agent in self.agents:
            ax, ay = agent.pos
            grid[ax, ay] = str(agent.id)

        out = ""
        for row in grid.T:
            out += " ".join(row) + "\n"
        return out
