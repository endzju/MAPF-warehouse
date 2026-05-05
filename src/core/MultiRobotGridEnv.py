from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from src.agents.delivery_robot import DeliveryRobot
from src.agents.depot import Depot
from src.agents.task import Task
from src.utils.enums import TaskType


class MultiRobotGridEnv(gym.Env):
    def __init__(
        self,
        grid_size: tuple[int, int] = (10, 10),
        num_agents: int = 5,
        agent_view_size: int = 5,
        obstacles: set[tuple[int, int]] | None = None,
        depots: list[Depot] = [Depot((0, 0))],
        step_limit: int = 100,
        task_length: int = 5,
    ):
        super(MultiRobotGridEnv, self).__init__()
        self.grid_width, self.grid_height = grid_size
        self.num_states = self.grid_width * self.grid_height
        self.num_agents = num_agents
        self.obstacles = np.zeros((self.grid_width, self.grid_height), dtype=np.uint8)
        self.obstacle_set = set()
        self.agent_view_size = agent_view_size
        self.agents: set[DeliveryRobot] = set()

        if obstacles:
            obs = np.array(obstacles, dtype=np.uint8)
            self.obstacles[obs[:, 0], obs[:, 1]] = 1
            self.obstacle_set = obstacles

        self.depots = depots
        self.step_limit = step_limit
        self.step_count = 0
        self.task_length = task_length

        # Akcje: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
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
                # If hit wall or other agent
                "aditional_info": spaces.Box(
                    low=0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        # Visualization
        self.window_size = 800  # Rozmiar okna w pikselach
        self.cell_size = self.window_size // max(self.grid_width, self.grid_height)
        self.window = None
        self.clock = None
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 18)

    def _depot_positions(self) -> set[tuple[int, int]]:
        return {depot.pos for depot in self.depots}

    def _obstacle_cells(self) -> set[tuple[int, int]]:
        indices = np.argwhere(self.obstacles == 1)
        return set(map(tuple, indices))

    def get_empty_cells(self, is_depot_obstacle: bool = False) -> set[tuple[int, int]]:
        occupied_cells = self._obstacle_cells()
        for agent in self.agents:
            occupied_cells.update(agent.get_occupied_cells())
        if is_depot_obstacle:
            occupied_cells.update(self._depot_positions())
        all_cells = {
            (x, y) for x in range(self.grid_width) for y in range(self.grid_height)
        }
        empty_cells = all_cells - occupied_cells
        return empty_cells

    def reset(self, seed=None, options=None):
        self.step_count = 0
        super().reset(seed=seed)
        self._calc_padded_obstacle_grid()
        self.agents.clear()
        empty_cells = list(self.get_empty_cells(is_depot_obstacle=True))
        agent_indices = self.np_random.choice(
            len(empty_cells), size=self.num_agents, replace=False
        )
        goals = []
        for i in range(self.num_agents):
            goals.append(
                self.np_random.choice(
                    len(empty_cells), size=self.task_length, replace=True
                )
            )
        observations = {}

        for i, (agent_i, goal_indices) in enumerate(zip(agent_indices, goals)):
            agent_pos = empty_cells[agent_i]
            goal_positions = [empty_cells[goal_i] for goal_i in goal_indices] + [
                self.depots[0].pos
            ]
            task_types = [TaskType.PICKUP] * len(goal_indices) + [TaskType.LEAVE]
            task = Task(goal_positions, task_types, id=i)
            robot = DeliveryRobot(
                position=agent_pos, task=task, depot=self.depots[0], id=i
            )
            self.agents.add(robot)
            observations[robot.id] = self._get_obs(robot)

        return observations, {}

    def _next_pos(self, agent: DeliveryRobot, action: int) -> tuple[int, int]:
        dx, dy = 0, 0
        match action:
            case 0:  # UP
                dy = -1
            case 1:  # RIGHT
                dx = 1
            case 2:  # DOWN
                dy = 1
            case 3:  # LEFT
                dx = -1
        return (agent.pos[0] + dx, agent.pos[1] + dy)

    def step(self, actions: dict[int, int]):
        # actions to array np. [akcja_robota_0, akcja_robota_1, ...]
        observations = {}
        rewards = {}
        self._calc_padded_obstacle_grid()

        previous_empty_cells = self.get_empty_cells(is_depot_obstacle=False)
        empty_cells = set(previous_empty_cells)
        agent_list = list(self.agents)
        self.np_random.shuffle(agent_list)

        for agent in agent_list:
            if agent.id in actions:
                next_pos = self._next_pos(agent, actions[agent.id])
                if next_pos in empty_cells:
                    rewards[agent.id] = agent.reward(next_pos, empty_cells)
                    agent.set_next_pos(next_pos)
                    empty_cells.remove(next_pos)
                else:
                    rewards[agent.id] = agent.reward(next_pos, previous_empty_cells)

        remove_agents = set()

        for agent in self.agents:
            if agent.step():
                remove_agents.add(agent)
            elif not agent.is_idle() and not agent.is_done():
                observations[agent.id] = self._get_obs(agent)

        self.agents.difference_update(remove_agents)

        truncated = self.step_count >= self.step_limit
        terminated = len(self.agents) == 0

        self.step_count += 1
        return observations, rewards, terminated, truncated, {}

    def _in_view(self, agent: DeliveryRobot, pos: tuple[int, int]) -> bool:
        if agent.pos is None:
            raise ValueError("Agent has no position")
        elif pos is None:
            raise ValueError("View position is None")
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
        return (pos[1] - ay + radius, pos[0] - ax + radius)

    def _add_if_in_view(
        self, agent: DeliveryRobot, pos: tuple[int, int], view: np.ndarray
    ):
        if pos is None:
            return
        if self._in_view(agent, pos):
            view_pos = self._view_position(agent, pos)
            view[view_pos] = 1

    def _goal_vector(self, agent: DeliveryRobot) -> np.ndarray:
        if agent.goal_pos is None:
            raise ValueError("Agent has no goal")
        max_dist = max(self.grid_width, self.grid_height)

        dx = (agent.goal_pos[0] - agent.pos[0]) / max_dist
        if dx < 0:
            dx = -1
        elif dx > 0:
            dx = 1
        dy = (agent.goal_pos[1] - agent.pos[1]) / max_dist
        if dy < 0:
            dy = -1
        elif dy > 0:
            dy = 1

        goal_vec = [dx, dy]
        return np.array(goal_vec, dtype=np.float32)

    def _calc_padded_obstacle_grid(self):
        radius = self.agent_view_size // 2
        self._padded_obstacle_grid = np.pad(
            self.obstacles, pad_width=radius, mode="constant", constant_values=1
        )

    def _get_obs(self, agent: DeliveryRobot) -> dict[str, np.ndarray]:
        radius = self.agent_view_size // 2

        ax, ay = agent.pos
        padded_ax = ax + radius
        padded_ay = ay + radius
        obstacles = self._padded_obstacle_grid[
            padded_ay - radius : padded_ay + radius + 1,
            padded_ax - radius : padded_ax + radius + 1,
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

            self._add_if_in_view(agent, other_agent.goal_pos, other_agent_goals)

        self._add_if_in_view(agent, agent.goal_pos, agent_goal)

        goal_vector = self._goal_vector(agent)

        return {
            "view": np.stack(
                [obstacles, other_agent_positions, other_agent_goals, agent_goal]
            ),
            "goal_vector": goal_vector,
            "aditional_info": 0,
        }

    def render_as_text(self, mode="human") -> str:
        grid = np.full((self.grid_width, self.grid_height), ".", dtype=str)
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.obstacles[x, y]:
                    grid[x, y] = "#"

        for d in self.depots:
            gx, gy = d.pos
            grid[gx, gy] = "D"
        for agent in self.agents:
            if agent.task_type == TaskType.PICKUP:
                ax, ay = agent.goal_pos
                grid[ax, ay] = "o"
        for agent in self.agents:
            ax, ay = agent.pos
            grid[ax, ay] = str(agent.id)

        out = ""
        for row in grid.T:
            out += " ".join(row) + "\n"
        return out

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size), pygame.RESIZABLE
            )
            pygame.display.set_caption("Multi-Robot Delivery Grid")
            images_path = Path(__file__).parent.parent / "assets" / "images"
            self.robot_img = pygame.image.load(
                images_path / "robot.png"
            ).convert_alpha()
            self.package_img = pygame.image.load(
                images_path / "package.png"
            ).convert_alpha()
            self.exit_img = pygame.image.load(images_path / "exit.png").convert_alpha()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # Białe tło
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                # Aktualizacja rozmiaru okna i powierzchni
                self.window = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )

        current_w, current_h = self.window.get_size()
        canvas = pygame.Surface((current_w, current_h))
        canvas.fill((255, 255, 255))

        scale_x = current_w // self.grid_width
        scale_y = current_h // self.grid_height
        dynamic_cell_size = min(scale_x, scale_y)

        grid_total_width = self.grid_width * dynamic_cell_size
        grid_total_height = self.grid_height * dynamic_cell_size

        # Obliczamy marginesy, aby wyśrodkować siatkę
        offset_x = (current_w - grid_total_width) // 2
        offset_y = (current_h - grid_total_height) // 2

        # 1. Rysowanie siatki i przeszkód
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(
                    offset_x + x * dynamic_cell_size,
                    offset_y + y * dynamic_cell_size,
                    dynamic_cell_size,
                    dynamic_cell_size,
                )

                # Przeszkody (Czarne)
                if self.obstacles[x, y]:
                    pygame.draw.rect(canvas, (40, 40, 40), rect)

                # Siatka (Szare linie)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)

        # 2. Rysowanie Depotów (Niebieskie kwadraty)
        for depot in self.depots:
            d_rect = pygame.Rect(
                offset_x + depot.pos[0] * dynamic_cell_size,
                offset_y + depot.pos[1] * dynamic_cell_size,
                dynamic_cell_size,
                dynamic_cell_size,
            )
            pygame.draw.rect(canvas, (0, 0, 255), d_rect)

        # 3. Rysowanie Robotów i ich Celów
        for agent in self.agents:
            text_color = (0, 0, 0)
            agent_id_str = f"ID:{agent.id}"
            # Cel (Małe kółko w kolorze agenta, ale przezroczyste/jasne)
            if agent.goal_pos:
                pos_x = offset_x + agent.goal_pos[0] * dynamic_cell_size
                pos_y = offset_y + agent.goal_pos[1] * dynamic_cell_size
                goal_img = (
                    self.exit_img
                    if agent.task_type == TaskType.LEAVE
                    else self.package_img
                )
                goal_scaled = pygame.transform.scale(
                    goal_img, (dynamic_cell_size, dynamic_cell_size)
                )

                r = ((agent.id + 1) * 50) % 256
                g = ((agent.id + 1) * 80) % 256
                b = ((agent.id + 1) * 110) % 256
                goal_scaled.fill((r, g, b, 255), special_flags=pygame.BLEND_RGBA_MULT)

                canvas.blit(goal_scaled, (pos_x, pos_y))

                goal_text = self.font.render(f"G:{agent.id}", True, text_color)
                canvas.blit(goal_text, (pos_x, pos_y - 15))

            # 1. Pozycja
            pos_x = offset_x + agent.pos[0] * dynamic_cell_size
            pos_y = offset_y + agent.pos[1] * dynamic_cell_size

            # 2. Skalowanie
            robot_scaled = pygame.transform.scale(
                self.robot_img, (dynamic_cell_size, dynamic_cell_size)
            )

            # 3. Kolorowanie (opcjonalnie, jeśli obrazek jest biały)
            r = ((agent.id + 1) * 50) % 256
            g = ((agent.id + 1) * 80) % 256
            b = ((agent.id + 1) * 110) % 256
            robot_scaled.fill((r, g, b, 255), special_flags=pygame.BLEND_RGBA_MULT)

            # 4. Wyświetlenie
            canvas.blit(robot_scaled, (pos_x, pos_y))

            # 5. Tekst ID (centrowanie napisu nad robotem)
            agent_id_str = f"ID:{agent.id}"
            agent_text = self.font.render(agent_id_str, True, (0, 0, 0))
            canvas.blit(agent_text, (pos_x, pos_y - 15))

        # Wyświetlenie na ekranie
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # Ograniczenie FPS (np. 10 klatek na sekundę)
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
