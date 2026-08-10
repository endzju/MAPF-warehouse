import ctypes
import heapq
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.agents.delivery_robot import DeliveryRobot
from src.models.depot import Depot
from src.models.task import Task
from src.utils.distance import manhattan_distance
from src.utils.enums import TaskType

ctypes.windll.shcore.SetProcessDpiAwareness(1)

import time

import pygame


class MultiRobotGridEnv(gym.Env):
    def __init__(
        self,
        grid_size: tuple[int, int] = (20, 20),
        agent_view_size: int = 5,
        obstacles: set[tuple[int, int]] | None = None,
        input_depots: list[Depot] | None = None,
        output_depots: list[Depot] | None = None,
        step_limit: int = 100,
        task_length: int = 5,
        finish_times: dict[TaskType, int] | None = None,
        tasks: list[Task] | None = None,
        num_tasks: int = 100,
        max_robots: int = 100,
        modulo_reward_x: tuple[int] = (),
        modulo_reward_y: tuple[int] = (),
        modulo_goal_distance_reward_cancel: int = 5,
        float_goal_vector: bool = False,
        view_dims: int = 4,
        goal_vec_size: int = 2,
        n_actions: int = 5,
    ):
        super().__init__()
        self.grid_width, self.grid_height = grid_size
        self.obstacles = np.zeros((self.grid_width, self.grid_height), dtype=np.uint8)
        self.obstacle_set = set()
        self.agent_view_size = agent_view_size
        self.agents: set[DeliveryRobot]
        self.avg_manhattan_distance = 0
        self.avg_delivery_time = 0
        self.deliveries = 0
        self.finish_times = finish_times or {
            TaskType.ENTER: 1,
            TaskType.PICKUP: 1,
            TaskType.LEAVE: 1,
        }
        self.given_tasks = tasks or []
        self.tasks = []
        self.num_tasks = num_tasks
        self.max_robots = max_robots
        self.modulo_reward_x = modulo_reward_x
        self.modulo_reward_y = modulo_reward_y
        self.modulo_goal_distance_reward_cancel = modulo_goal_distance_reward_cancel
        self.float_goal_vector = float_goal_vector
        self.view_dims = view_dims
        self.goal_vec_size = goal_vec_size
        self.n_actions = n_actions
        self.depot_priority = 0
        self.deleted_agents = []

        if obstacles:
            obs = np.array(list[tuple[int, int]](obstacles), dtype=np.intp)
            self.obstacles[obs[:, 0], obs[:, 1]] = 1
            self.obstacle_set = set[tuple[int, int]](obstacles)

        self.empty_obs = np.zeros(
            (self.agent_view_size, self.agent_view_size), dtype=np.uint8
        )

        self.modulo_x_size = len(self.modulo_reward_x)
        self.modulo_y_size = len(self.modulo_reward_y)
        self.additional_input_size = (
            self.goal_vec_size + self.modulo_x_size + self.modulo_y_size
        )

        self.input_depots = input_depots or self._default_input_depots()
        self.output_depots = output_depots or self._default_output_depots()
        self._reset_depot_max_robots()
        if len(self.input_depots) != len(self.output_depots):
            raise ValueError(
                "Number of input depots must be equal to number of output depots"
            )
        self.step_limit = step_limit
        self.step_count = 0
        self.task_length = task_length
        self.id_counter = 0
        self.paused = False

        # Akcje: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
        self.action_space = spaces.MultiDiscrete([self.n_actions] * self.max_robots)

        self.observation_space = spaces.Dict(
            {
                # obstacles, other agent pos, other agnet goal pos, agent goal pos
                "view": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        self.view_dims,
                        self.agent_view_size,
                        self.agent_view_size,
                    ),
                    dtype=np.uint8,
                ),
                # difference in x and y coordinates from the goal
                "additional_input": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.max_robots, 2),
                    dtype=np.float32,
                ),
            }
        )

        # Visualization
        self.window_size = 900  # Rozmiar okna w pikselach
        self.cell_size = self.window_size // max(self.grid_width, self.grid_height)
        self.window = None
        self.clock = None
        self.path_length = 20
        self.font = None
        self.info_font = None
        self.alert_font = None
        self._static_layer = None
        self._static_layer_key = None
        self._sprite_cache = {}
        self._text_cache = {}

    def _get_unique_depots(self) -> list[Depot]:
        return list(set(self.input_depots + self.output_depots))

    def _depot_positions(self) -> set[tuple[int, int]]:
        return {depot.pos for depot in self._get_unique_depots()}

    def _default_input_depots(self):
        return [
            Depot((0, 0)),
            Depot((self.grid_width - 1, 0)),
        ]

    def _default_output_depots(self):
        return [
            Depot((0, self.grid_height - 1)),
            Depot((self.grid_width - 1, self.grid_height - 1)),
        ]

    def _obstacle_cells(self) -> set[tuple[int, int]]:
        indices = np.argwhere(self.obstacles == 1)
        return set(map(tuple, indices))

    def _reset_depot_max_robots(self):
        for i in range(len(self.input_depots)):
            self.input_depots[i].stored_robots = self.max_robots // len(
                self.input_depots
            )

    def _calc_grid_caches(self):
        """
        Rebuild everything derived from the grid size, the obstacles and the view
        size. Must be re-run whenever any of those change, since `agent_view_size`
        and the obstacle layout may be reassigned after construction.
        """
        radius = self.agent_view_size // 2
        self.view_radius = radius
        self.view_window = 2 * radius + 1
        self.max_dist = max(self.grid_width, self.grid_height)

        self._padded_obstacle_grid = np.pad(
            self.obstacles, pad_width=radius, mode="constant", constant_values=1
        )
        self._obstacle_cells_cache = self._obstacle_cells()
        self._all_cells = {
            (x, y) for x in range(self.grid_width) for y in range(self.grid_height)
        }

        # Scratch occupancy grids for the agent and goal view channels, stacked so
        # both are windowed in one operation and indexed as [y, x] to match the
        # row/column order the observation windows are written in.
        self._view_grids = np.zeros(
            (
                2,
                self.grid_width + 2 * radius,
                self.grid_height + 2 * radius,
            ),
            dtype=np.uint8,
        )

    def get_empty_cells(
        self, include_depot: bool = False, radius_from_depot: int = 0
    ) -> set[tuple[int, int]]:
        occupied_cells = set(self._obstacle_cells_cache)
        for agent in self.agents:
            occupied_cells.update(agent.get_occupied_cells())
        if include_depot:
            occupied_cells.update(self._depot_positions())
        if radius_from_depot:
            for depot_pos in self._depot_positions():
                for dx in range(-radius_from_depot, radius_from_depot + 1):
                    for dy in range(-radius_from_depot, radius_from_depot + 1):
                        occupied_cells.add((depot_pos[0] + dx, depot_pos[1] + dy))

        return self._all_cells - occupied_cells

    def get_agent_positions(
        self, include_next_pos: bool = True
    ) -> set[tuple[int, int]]:
        occupied_cells = set()
        for agent in self.agents:
            occupied_cells.add(agent.pos)
            if include_next_pos and agent.next_pos is not None:
                occupied_cells.add(agent.next_pos)
        return occupied_cells

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.avg_manhattan_distance = 0
        self.avg_delivery_time = 0
        self.deliveries = 0
        self.depot_priority = 0
        self.tasks = []
        super().reset(seed=seed)
        self._calc_grid_caches()
        self.agents = set()
        self.available_ids = deque(range(self.max_robots))
        empty_cells = list(
            self.get_empty_cells(include_depot=False, radius_from_depot=1)
        )
        self._reset_depot_max_robots()
        for depot in self.input_depots:
            depot._clear_tasks()

        if self.given_tasks:
            self.tasks = self.given_tasks
        else:
            goal_indices = self.np_random.integers(
                0, len(empty_cells), size=(self.num_tasks, self.task_length)
            )
            for i in range(self.num_tasks):
                goal_positions = [empty_cells[j] for j in goal_indices[i]]
                goal_types = [TaskType.PICKUP] * len(goal_positions)
                task = Task(goal_positions=goal_positions, goal_types=goal_types, id=i)
                self.tasks.append(task)

        for idx, task in enumerate(self.tasks):
            task.reset()
            depot = self.input_depots[idx % len(self.input_depots)]
            depot.add_task(task)

        observations = {}
        self.avg_manhattan_distance = self._avg_manhattan_time()
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

    def _next_id(self):
        self.id_counter += 1
        return self.id_counter - 1

    def _all_tasks_completed(self):
        if self.agents:
            return False
        for depot in self.input_depots:
            if len(depot.tasks) > 0:
                return False
        return True

    def get_num_tasks_completed(self) -> int:
        completed = 0
        for depot in self.input_depots:
            completed += len(depot.finished_tasks)
        return completed

    def reward(
        self,
        agent: DeliveryRobot,
        next_pos: tuple[int, int],
        blocked_cells: set[tuple[int, int]],
    ):
        reward = -1
        if next_pos == agent.pos:
            return reward
        x, y = next_pos
        if (
            not (0 <= x < self.grid_width and 0 <= y < self.grid_height)
            or next_pos in blocked_cells
        ):
            return -20
        if next_pos == agent.goal_pos:
            return 100

        old_dist = agent._goal_distance(agent.pos)
        new_dist = agent._goal_distance(next_pos)
        dist_reward = old_dist - new_dist

        if dist_reward < 0:
            dist_reward *= 5
        else:
            dist_reward *= 2.5
        reward += dist_reward

        allow_modulo_reward = (
            abs(agent.pos[0] - agent.goal_pos[0])
            > self.modulo_goal_distance_reward_cancel
            and abs(agent.pos[1] - agent.goal_pos[1])
            > self.modulo_goal_distance_reward_cancel
        )

        if allow_modulo_reward:
            if self.modulo_reward_x:
                dx = next_pos[0] - agent.pos[0]
                reward += self.modulo_reward_x[agent.pos[0] % self.modulo_x_size] * dx
            if self.modulo_reward_y:
                dy = next_pos[1] - agent.pos[1]
                reward += self.modulo_reward_y[agent.pos[1] % self.modulo_y_size] * dy
        return reward

    def _is_inside_grid(self, pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_width and 0 <= pos[1] < self.grid_height

    def _step_agents(self, actions, previous_blocked) -> dict[int, float]:
        rewards = {}

        blocked = set(previous_blocked)

        agent_list = list(self.agents)
        random.shuffle(agent_list)

        for agent in agent_list:
            if agent.is_idle():
                continue
            action = actions.get(agent.id)
            if action is None:
                raise ValueError("Agent has no action")

            if agent.is_stuck():
                rewards[agent.id] = 0
                random_move = random.randrange(self.n_actions)
                next_pos = self._next_pos(agent, random_move)
                if self._is_inside_grid(next_pos) and next_pos not in blocked:
                    agent.set_next_pos(next_pos)
                    blocked.add(next_pos)
                continue

            next_pos = self._next_pos(agent, action)
            if self._is_inside_grid(next_pos) and next_pos not in blocked:
                rewards[agent.id] = self.reward(agent, next_pos, blocked)
                agent.set_next_pos(next_pos)
                blocked.add(next_pos)
            else:
                rewards[agent.id] = self.reward(agent, next_pos, previous_blocked)

        remove_agents = set()
        # Move agents and remove those that are done
        for agent in self.agents:
            if agent.step():
                self.avg_delivery_time = (
                    self.avg_delivery_time * self.deliveries + agent.step_count
                ) / (self.deliveries + 1)
                self.deliveries += 1
                remove_agents.add(agent)
                agent.in_depot.stored_robots += 1
        self.agents -= remove_agents
        self.deleted_agents += list(remove_agents)

        return rewards

    def _deploy_robots(self, previous_blocked):
        # deploy robot if tasks left
        for in_depot, out_depot in zip(self.input_depots, self.output_depots):
            if (
                in_depot.pos not in previous_blocked
                and in_depot.stored_robots > 0
                and in_depot.has_tasks()
            ):
                if len(self.deleted_agents) > 0:
                    agent = self.deleted_agents.pop()
                    agent.reset()
                    agent.pos = in_depot.pos
                    agent.task = in_depot.pop_task()
                    agent.in_depot = in_depot
                    agent.out_depot = out_depot
                    agent.id = self._next_id()
                else:
                    agent = DeliveryRobot(
                        position=in_depot.pos,
                        task=in_depot.pop_task(),
                        in_depot=in_depot,
                        out_depot=out_depot,
                        id=self._next_id(),
                    )
                agent.idle_time = self.finish_times[TaskType.ENTER]
                self.agents.add(agent)
                in_depot.stored_robots -= 1

    def _get_observations(self) -> dict[int, np.ndarray]:
        observations = {}
        agent_positions = {agent.pos for agent in self.agents}
        agent_goal_positions = [
            agent.goal_pos for agent in self.agents if agent.goal_pos is not None
        ]
        view_grids = self._build_view_grids(agent_positions, agent_goal_positions)

        for agent in self.agents:
            if not agent.is_idle():
                observations[agent.id] = self._get_obs(
                    agent=agent,
                    view_grids=view_grids,
                )

        return observations

    def step(self, actions: dict[int, int]):
        previous_blocked = self._obstacle_cells_cache | self.get_agent_positions(
            include_next_pos=True
        )
        rewards = self._step_agents(actions=actions, previous_blocked=previous_blocked)
        self._deploy_robots(previous_blocked=previous_blocked)
        observations = self._get_observations()

        self.step_count += 1

        truncated = self.step_count >= self.step_limit
        terminated = self._all_tasks_completed()

        return observations, rewards, terminated, truncated, {}

    def _goal_vector(self, agent: DeliveryRobot) -> np.ndarray:
        if agent.goal_pos is None:
            raise ValueError("Agent has no goal")

        gx, gy = agent.goal_pos
        ax, ay = agent.pos

        dx = (gx > ax) - (gx < ax)
        dy = (gy > ay) - (gy < ay)

        if self.float_goal_vector:
            dx = dx / self.max_dist
            dy = dy / self.max_dist

        goal_vector = np.zeros(self.goal_vec_size, dtype=np.float32)
        goal_vector[0] = dx
        goal_vector[1] = dy
        return goal_vector

    def _build_view_grids(
        self,
        agent_positions: set[tuple[int, int]],
        agent_goal_positions: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Stamp every agent position and every agent goal onto the padded scratch
        grids once per step, so each observation window is a plain array slice
        instead of a per-agent set intersection.
        """
        pad = self.view_radius
        grids = self._view_grids
        grids.fill(0)

        if agent_positions:
            coords = np.array(list(agent_positions), dtype=np.intp)
            grids[0, coords[:, 0] + pad, coords[:, 1] + pad] = 1
        if agent_goal_positions:
            coords = np.array(agent_goal_positions, dtype=np.intp)
            grids[1, coords[:, 0] + pad, coords[:, 1] + pad] = 1

        return grids

    def _get_obs(
        self,
        agent: DeliveryRobot,
        view_grids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        view = np.zeros(
            (self.view_dims, self.agent_view_size, self.agent_view_size), dtype=np.uint8
        )
        # chanel 0: obstacle grid
        # chanel 1: other agents
        # chanel 2: other agents' goals
        # chanel 3: own goal

        radius = self.view_radius
        window = self.view_window
        ax, ay = agent.pos
        gx, gy = agent.goal_pos

        view[0] = self._padded_obstacle_grid[ax : ax + window, ay : ay + window]
        view[1:3] = view_grids[:, ax : ax + window, ay : ay + window]
        view[1, radius, radius] = 0

        gdx = gx - ax
        gdy = gy - ay
        if -radius <= gdx <= radius and -radius <= gdy <= radius:
            # Likewise the agent's own goal belongs in channel 3, not channel 2.
            view[2, gdy + radius, gdx + radius] = 0
            view[3, gdy + radius, gdx + radius] = 1

        additional_input = np.zeros(self.additional_input_size, dtype=np.float32)
        goal_vec_size = self.goal_vec_size
        additional_input[:goal_vec_size] = self._goal_vector(agent=agent)

        modulo_x_size = self.modulo_x_size
        modulo_y_size = self.modulo_y_size
        if modulo_x_size:
            additional_input[goal_vec_size + ax % modulo_x_size] = 1
        if modulo_y_size:
            additional_input[goal_vec_size + modulo_x_size + ay % modulo_y_size] = 1

        return {
            "view": view,
            "additional_input": additional_input,
        }

    def _avg_manhattan_time(self) -> float:
        out_time_dict = defaultdict(list)
        delivery_time = 0

        # Calculate manhattan time of all tasks of all depots
        for in_depot, out_depot in zip(self.input_depots, self.output_depots):
            exit_times_heap = []
            exit_times = []
            time_from_start = 0
            for task in in_depot.tasks:
                # waiting to enter
                positions = [in_depot.pos] + task.goal_positions + [out_depot.pos]
                time = (
                    len(task.goal_positions) * self.finish_times[TaskType.PICKUP]
                    + self.finish_times[TaskType.ENTER]
                    + self.finish_times[TaskType.LEAVE]
                )
                for i in range(len(positions) - 1):
                    time += manhattan_distance(positions[i], positions[i + 1])
                delivery_time += time
                if len(exit_times_heap) == in_depot.stored_robots:
                    time_from_start = max(
                        heapq.heappop(exit_times_heap), time_from_start
                    )
                exit_time = time_from_start + time
                heapq.heappush(exit_times_heap, exit_time)
                exit_times.append(exit_time)
                time_from_start += 1 + self.finish_times[TaskType.ENTER]

            out_time_dict[out_depot] += exit_times

        # include waiting in queue to depot

        for time_list in out_time_dict.values():
            if not time_list:
                continue
            time_list.sort()
            min_dist = time_list[0]
            for i in range(len(time_list)):
                new_dist = max(min_dist, time_list[i])
                diff = new_dist - time_list[i]
                delivery_time += diff
                time_list[i] = new_dist
                min_dist = time_list[i] + 1 + self.finish_times[TaskType.LEAVE]
        return delivery_time / len(self.tasks)

    def handle_events(self):
        events = pygame.event.get()

        quit_requested = False
        pause_pressed = False

        for event in events:
            if event.type == pygame.QUIT:
                quit_requested = True

            elif event.type == pygame.VIDEORESIZE:
                self.window = pygame.display.set_mode(
                    (event.w, event.h),
                    pygame.RESIZABLE,
                )

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause_pressed = True
                if event.key == pygame.K_EQUALS:
                    self.path_length += 10
                if event.key == pygame.K_MINUS:
                    self.path_length -= 10
                    self.path_length = max(5, self.path_length)

            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.path_length += 5
                elif event.y < 0:
                    self.path_length -= 5
                    self.path_length = max(5, self.path_length)

        return quit_requested, pause_pressed

    def render_move(self, move_time: float = 1, fps: int = 10):
        frames = max(1, round(fps * move_time))

        for frame_num in range(1, frames + 1):
            self.render(animation_progress=frame_num / frames)
            time.sleep(move_time / frames)

    def _ensure_display(self):
        if self.window is not None:
            return

        if sys.platform == "win32":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(
            (self.window_size * 2, self.window_size), pygame.RESIZABLE
        )
        pygame.display.set_caption("Multi-Robot Delivery Grid")
        images_path = Path(__file__).parent.parent / "assets" / "images"
        self.robot_img = pygame.image.load(images_path / "robot.png").convert_alpha()
        self.package_img = pygame.image.load(
            images_path / "package.png"
        ).convert_alpha()
        self.exit_img = pygame.image.load(images_path / "exit.png").convert_alpha()

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 18)
        self.info_font = pygame.font.SysFont("Arial", 22)
        self.alert_font = pygame.font.SysFont("Arial", 32)

    def _agent_color(self, agent_id: int) -> tuple[int, int, int]:
        return (
            ((agent_id + 1) * 50) % 256,
            ((agent_id + 1) * 80) % 256,
            ((agent_id + 1) * 110) % 256,
        )

    def _static_layer_surface(
        self,
        size: tuple[int, int],
        offset_x: int,
        offset_y: int,
        cell_size: int,
    ) -> pygame.Surface:
        """
        The grid lines, obstacles and depots never move, so they are drawn once
        per window geometry instead of once per frame.
        """
        key = (size, offset_x, offset_y, cell_size)
        if self._static_layer_key == key:
            return self._static_layer

        surface = pygame.Surface(size)
        surface.fill((255, 255, 255))

        # 1. Rysowanie siatki i przeszkód
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(
                    offset_x + x * cell_size,
                    offset_y + y * cell_size,
                    cell_size,
                    cell_size,
                )

                # Przeszkody (Czarne)
                if self.obstacles[x, y]:
                    pygame.draw.rect(surface, (40, 40, 40), rect)

                # Siatka (Szare linie)
                pygame.draw.rect(surface, (200, 200, 200), rect, 1)

        # 2. Rysowanie Depotów (Niebieskie kwadraty)
        for depot in self._get_unique_depots():
            d_rect = pygame.Rect(
                offset_x + depot.pos[0] * cell_size,
                offset_y + depot.pos[1] * cell_size,
                cell_size,
                cell_size,
            )
            pygame.draw.rect(surface, (0, 0, 255), d_rect)

        self._static_layer = surface
        self._static_layer_key = key
        return surface

    def _tinted_sprite(
        self,
        image: pygame.Surface,
        name: str,
        cell_size: int,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        key = (name, cell_size, color)
        sprite = self._sprite_cache.get(key)
        if sprite is None:
            sprite = pygame.transform.scale(image, (cell_size, cell_size))
            sprite.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MULT)
            if len(self._sprite_cache) > 2048:
                self._sprite_cache.clear()
            self._sprite_cache[key] = sprite
        return sprite

    def _label(self, text: str, color: tuple[int, int, int]) -> pygame.Surface:
        key = (text, color)
        label = self._text_cache.get(key)
        if label is None:
            label = self.font.render(text, True, color)
            if len(self._text_cache) > 4096:
                self._text_cache.clear()
            self._text_cache[key] = label
        return label

    def render(self, paused: bool = False, animation_progress: float = 1):

        self._ensure_display()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        pygame.event.pump()

        current_w, current_h = self.window.get_size()

        scale_x = current_w // self.grid_width
        scale_y = current_h // self.grid_height
        dynamic_cell_size = min(scale_x, scale_y)

        grid_total_width = self.grid_width * dynamic_cell_size
        grid_total_height = self.grid_height * dynamic_cell_size

        # Obliczamy marginesy, aby wyśrodkować siatkę
        offset_x = (current_w - grid_total_width) // 4
        offset_y = (current_h - grid_total_height) // 2

        canvas = self._static_layer_surface(
            (current_w, current_h), offset_x, offset_y, dynamic_cell_size
        ).copy()

        # 3. Rysowanie Robotów i ich Celów
        text_color = (0, 0, 0)

        # Drawing packages and depots
        for agent in self.agents:
            if agent.goal_pos:
                pos_x = offset_x + agent.goal_pos[0] * dynamic_cell_size
                pos_y = offset_y + agent.goal_pos[1] * dynamic_cell_size
                is_exit = agent.task_type == TaskType.LEAVE
                goal_img = self.exit_img if is_exit else self.package_img

                goal_scaled = self._tinted_sprite(
                    goal_img,
                    "exit" if is_exit else "package",
                    dynamic_cell_size,
                    self._agent_color(agent.id),
                )

                canvas.blit(goal_scaled, (pos_x, pos_y))

                goal_text = self._label(f"G:{agent.id}", text_color)
                canvas.blit(goal_text, (pos_x, pos_y - 15))

        mouse_x, mouse_y = pygame.mouse.get_pos()
        hovered_agent = None

        # Drawing robots
        for agent in self.agents:
            if len(agent.pos_history) == 0 or agent.pos_history[-1] == agent.pos:
                moving_pos_x, moving_pos_y = agent.pos[0], agent.pos[1]
            else:
                dx = agent.pos[0] - agent.pos_history[-1][0]
                dy = agent.pos[1] - agent.pos_history[-1][1]
                moving_pos_x = agent.pos_history[-1][0] + animation_progress * dx
                moving_pos_y = agent.pos_history[-1][1] + animation_progress * dy

            pos_x = offset_x + round(moving_pos_x * dynamic_cell_size)
            pos_y = offset_y + round(moving_pos_y * dynamic_cell_size)

            robot_rect = pygame.Rect(
                pos_x,
                pos_y,
                dynamic_cell_size,
                dynamic_cell_size,
            )

            if robot_rect.collidepoint(mouse_x, mouse_y):
                hovered_agent = agent

            robot_scaled = self._tinted_sprite(
                self.robot_img,
                "robot",
                dynamic_cell_size,
                self._agent_color(agent.id),
            )

            canvas.blit(robot_scaled, (pos_x, pos_y))

            agent_text = self._label(f"ID:{agent.id}", (0, 0, 0))
            canvas.blit(agent_text, (pos_x, pos_y - 15))

        text = f"Avg Manhattan Distance: {self.avg_manhattan_distance:.2f}"
        rendered_text = self.info_font.render(text, True, (0, 0, 0))
        canvas.blit(rendered_text, (current_w - 350, 20))

        delivery_text = (
            f"{self.avg_delivery_time:.2f}" if self.avg_delivery_time else "N/A"
        )
        text = f"Avg Delivery time: {delivery_text}"
        rendered_text = self.info_font.render(text, True, (0, 0, 0))
        canvas.blit(rendered_text, (current_w - 350, 50))
        if self.avg_delivery_time:
            text = f"Delivery efficiency: {100 * self.avg_manhattan_distance / self.avg_delivery_time:.2f}%"
        else:
            text = "Delivery efficiency: N/A"
        rendered_text = self.info_font.render(text, True, (0, 0, 0))
        canvas.blit(rendered_text, (current_w - 350, 80))

        if paused:
            text = "PAUSED"
            rendered_text = self.alert_font.render(text, True, (200, 0, 0))
            canvas.blit(rendered_text, (current_w - 200, current_h - 100))

        path_surface = pygame.Surface(
            (current_w, current_h),
            pygame.SRCALPHA,
        )

        if hovered_agent is not None:
            r, g, b = self._agent_color(hovered_agent.id)
            line_width = 10

            path = list(hovered_agent.pos_history) + [hovered_agent.pos]
            path = path[-self.path_length :]
            completed_goals = list(hovered_agent.task.get_delivered_goals())
            points = []
            completed_points = []

            for x, y in path:
                px = offset_x + x * dynamic_cell_size + dynamic_cell_size // 2
                py = offset_y + y * dynamic_cell_size + dynamic_cell_size // 2
                points.append((px, py))

            for x, y in completed_goals:
                px = offset_x + x * dynamic_cell_size + dynamic_cell_size // 2
                py = offset_y + y * dynamic_cell_size + dynamic_cell_size // 2
                completed_points.append((px, py))

            if len(points) >= 2:
                num_segments = len(points) - 1

                last_side = None
                cell_offset = dynamic_cell_size // 4

                entry_offset_dict = {
                    0: (cell_offset, cell_offset),
                    1: (cell_offset, -cell_offset),
                    2: (-cell_offset, -cell_offset),
                    3: (-cell_offset, cell_offset),
                }

                for i in range(num_segments):
                    start = points[i]
                    end = points[i + 1]
                    idx_start = path[i]
                    idx_end = path[i + 1]

                    dx = idx_end[0] - idx_start[0]
                    dy = idx_end[1] - idx_start[1]

                    alpha = int(255 * (i + 1) / num_segments)
                    segment_color = (r, g, b, alpha)

                    # right
                    if dx > 0:
                        start_rightsided = (
                            start[0] + cell_offset,
                            start[1] + cell_offset,
                        )
                        end_rightsided = (
                            end[0] - cell_offset,
                            end[1] + cell_offset,
                        )
                        cur_side = 0
                    # up
                    elif dy < 0:
                        start_rightsided = (
                            start[0] + cell_offset,
                            start[1] - cell_offset,
                        )
                        end_rightsided = (
                            end[0] + cell_offset,
                            end[1] + cell_offset,
                        )
                        cur_side = 1
                    # left
                    elif dx < 0:
                        start_rightsided = (
                            start[0] - cell_offset,
                            start[1] - cell_offset,
                        )
                        end_rightsided = (
                            end[0] + cell_offset,
                            end[1] - cell_offset,
                        )
                        cur_side = 2
                    # down
                    elif dy > 0:
                        start_rightsided = (
                            start[0] - cell_offset,
                            start[1] + cell_offset,
                        )
                        end_rightsided = (
                            end[0] - cell_offset,
                            end[1] - cell_offset,
                        )
                        cur_side = 3

                    if dx == 0 and dy == 0:
                        continue

                    if last_side is not None and last_side != cur_side:
                        while last_side != cur_side:
                            start_draw = (
                                start[0] + entry_offset_dict[last_side][0],
                                start[1] + entry_offset_dict[last_side][1],
                            )
                            last_side = (last_side + 1) % 4
                            end_draw = (
                                start[0] + entry_offset_dict[last_side][0],
                                start[1] + entry_offset_dict[last_side][1],
                            )

                            pygame.draw.line(
                                path_surface,
                                segment_color,
                                start_draw,
                                end_draw,
                                line_width,
                            )

                    last_side = (cur_side - 1) % 4

                    pygame.draw.line(
                        path_surface,
                        segment_color,
                        start_rightsided,
                        end_rightsided,
                        line_width,
                    )

                for point in completed_points:
                    pygame.draw.circle(path_surface, (255, 215, 0), point, line_width)

        # Wyświetlenie na ekranie
        canvas.blit(path_surface, (0, 0))
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.flip()

        # Ograniczenie FPS (np. 10 klatek na sekundę)
        self.clock.tick(20)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

            self.window = None
            self.clock = None
            self._static_layer = None
            self._static_layer_key = None
            self._sprite_cache.clear()
            self._text_cache.clear()
            self.font = None
            self.info_font = None
            self.alert_font = None
