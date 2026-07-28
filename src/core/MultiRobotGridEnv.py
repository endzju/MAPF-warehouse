import ctypes
import heapq
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
        input_depots: list[Depot] = [],
        output_depots: list[Depot] = [],
        step_limit: int = 100,
        task_length: int = 5,
        finish_times: dict[TaskType, int] | None = None,
        tasks: list[Task] = [],
        num_tasks: int = 100,
        max_robots: int = 100,
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
        self.given_tasks = tasks
        self.tasks = []
        self.num_tasks = num_tasks
        self.max_robots = max_robots
        self.depot_priority = 0
        self.float_goal_vec = False
        self.deleted_agents = []

        if obstacles:
            obs = np.array(obstacles, dtype=np.uint8)
            self.obstacles[obs[:, 0], obs[:, 1]] = 1
            self.obstacle_set = obstacles

        self.empty_obs = np.zeros(
            (self.agent_view_size, self.agent_view_size), dtype=np.uint8
        )

        self.input_depots = input_depots or self._default_input_depots()
        self.output_depots = output_depots or self._default_output_depots()
        self._update_depot_max_robots()
        if len(self.input_depots) != len(self.output_depots):
            raise ValueError(
                "Number of input depots must be equal to number of output depots"
            )
        self.step_limit = step_limit
        self.step_count = 0
        self.task_length = task_length
        self.id_counter = 0

        # Akcje: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
        # MultiDiscrete pozwala zdefiniować akcję dla każdego robota naraz
        self.action_space = spaces.MultiDiscrete([5] * self.max_robots)

        self.observation_space = spaces.Dict(
            {
                # obstacles, other agent pos, other agnet goal pos, agent goal pos
                "view": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        self.max_robots,
                        4,
                        self.agent_view_size,
                        self.agent_view_size,
                    ),
                    dtype=np.uint8,
                ),
                # difference in x and y coordinates from the goal
                "goal_vector": spaces.Box(
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
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 18)
        self.info_font = pygame.font.SysFont("Arial", 22)
        self.alert_font = pygame.font.SysFont("Arial", 32)

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

    def _increment_depot_priority(self):
        self.depot_priority = (self.depot_priority + 1) % len(self.input_depots)

    # def _get_priority_depot_indices(self) -> list[Depot]:
    #     depot_indices = []
    #     for i in range(len(self.input_depots)):
    #         depot_indices.append((i + self.depot_priority) % len(self.input_depots))

    def _update_depot_max_robots(self):
        for i in range(len(self.input_depots)):
            self.input_depots[i].stored_robots = self.max_robots // len(
                self.input_depots
            )

    def get_empty_cells(
        self, is_depot_obstacle: bool = False, depot_obstacle_radius: int = 0
    ) -> set[tuple[int, int]]:
        occupied_cells = self._obstacle_cells()
        for agent in self.agents:
            occupied_cells.update(agent.get_occupied_cells())
        if is_depot_obstacle:
            occupied_cells.update(self._depot_positions())
        if depot_obstacle_radius:
            for depot_pos in self._depot_positions():
                for dx in range(-depot_obstacle_radius, depot_obstacle_radius + 1):
                    for dy in range(-depot_obstacle_radius, depot_obstacle_radius + 1):
                        occupied_cells.add((depot_pos[0] + dx, depot_pos[1] + dy))

        all_cells = {
            (x, y) for x in range(self.grid_width) for y in range(self.grid_height)
        }
        empty_cells = all_cells - occupied_cells
        return empty_cells

    def get_all_occupied_cells(
        self, include_next_pos: bool = True
    ) -> set[tuple[int, int]]:
        occupied_cells = set()
        for agent in self.agents:
            occupied_cells.add(agent.pos)
            if include_next_pos and agent.next_pos is not None:
                occupied_cells.add(agent.next_pos)
        return occupied_cells

    def get_agent_view_cells(self, agent: DeliveryRobot):
        ax, ay = agent.pos
        cells = set()
        for dx in range(-self.agent_view_size // 2, self.agent_view_size // 2 + 1):
            for dy in range(-self.agent_view_size // 2, self.agent_view_size // 2 + 1):
                cells.add((ax + dx, ay + dy))
        return cells

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.avg_manhattan_distance = 0
        self.avg_delivery_time = 0
        self.deliveries = 0
        self.depot_priority = 0
        self.tasks = []
        super().reset(seed=seed)
        self._calc_padded_obstacle_grid()
        self.agents = set()
        self.available_ids = deque(range(self.max_robots))
        empty_cells = list(
            self.get_empty_cells(is_depot_obstacle=True, depot_obstacle_radius=2)
        )
        self._update_depot_max_robots()
        for depot in self.input_depots:
            depot._clear_tasks()

        if self.given_tasks:
            self.tasks = self.given_tasks
        else:
            for i in range(self.num_tasks):
                goal_positions = []
                goal_indices = self.np_random.choice(
                    len(empty_cells), size=self.task_length, replace=True
                )
                goal_positions += [empty_cells[i] for i in goal_indices]
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
            completed += len(depot.finnished_tasks)
        return completed

    time0 = 0
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 0

    def step(self, actions: list[int]):
        tic = time.time()
        # actions to array np. [akcja_robota_0, akcja_robota_1, ...]
        observations = {}
        rewards = {}

        previous_empty_cells = self.get_empty_cells(is_depot_obstacle=False)
        empty_cells = set(previous_empty_cells)
        agent_list = list(self.agents)
        self.np_random.shuffle(agent_list)

        self.time0 += time.time() - tic
        tic = time.time()

        for agent in agent_list:
            if agent.id in actions:
                if agent.is_stuck():
                    random_move = self.np_random.choice(4)
                    next_pos = self._next_pos(agent, random_move)
                    rewards[agent.id] = 0
                    if next_pos in empty_cells:
                        agent.set_next_pos(next_pos)
                        empty_cells.remove(next_pos)
                    continue

                next_pos = self._next_pos(agent, actions[agent.id])
                if next_pos in empty_cells:
                    rewards[agent.id] = agent.reward(next_pos, empty_cells)
                    agent.set_next_pos(next_pos)
                    empty_cells.remove(next_pos)
                else:
                    rewards[agent.id] = agent.reward(next_pos, previous_empty_cells)

        self.time1 += time.time() - tic
        tic = time.time()

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

        self.time2 += time.time() - tic
        tic = time.time()

        all_agent_positions = self.get_all_occupied_cells(include_next_pos=False)
        all_agent_goal_positions = {agent.goal_pos for agent in self.agents}

        for agent in self.agents:
            if not agent.is_done():
                observations[agent.id] = self._get_obs(
                    agent=agent,
                    all_agent_positions=all_agent_positions,
                    all_agent_goal_positions=all_agent_goal_positions,
                )
        truncated = self.step_count >= self.step_limit
        terminated = self._all_tasks_completed()

        self.step_count += 1
        agent_positions = {agent.pos for agent in self.agents}

        self.time3 += time.time() - tic
        tic = time.time()

        # deploy robot if tasks left
        for in_depot, out_depot in zip(self.input_depots, self.output_depots):
            if (
                in_depot.pos not in agent_positions
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
                self.agents.add(agent)
                in_depot.stored_robots -= 1

        self.time4 += time.time() - tic

        # randnum = self.np_random.random()
        # sum_time = self.time0 + self.time1 + self.time2 + self.time3 + self.time4
        # if randnum < 0.001 and sum_time > 0:
        #     print(
        #         f"time0: {self.time0 / sum_time} time1: {self.time1 / sum_time} time2: {self.time2 / sum_time} time3: {self.time3 / sum_time} time4: {self.time4 / sum_time}"
        #     )

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

        if self.float_goal_vec:
            dx = dx / max_dist
            dy = dy / max_dist

        goal_vec = [dx, dy]
        return np.array(goal_vec, dtype=np.float32)

    def _calc_padded_obstacle_grid(self):
        radius = self.agent_view_size // 2
        self._padded_obstacle_grid = np.pad(
            self.obstacles, pad_width=radius, mode="constant", constant_values=1
        )

    def _get_obs(
        self,
        agent: DeliveryRobot,
        all_agent_positions: set[tuple[int, int]],
        all_agent_goal_positions: set[tuple[int, int]],
        is_dummy=False,
    ) -> dict[str, np.ndarray]:
        view = np.zeros((4, self.agent_view_size, self.agent_view_size), dtype=np.uint8)
        radius = self.agent_view_size // 2

        if is_dummy:
            return {
                "view": np.zeros(
                    shape=(4, self.agent_view_size, self.agent_view_size),
                    dtype=np.uint8,
                ),
                "goal_vector": np.zeros(shape=(2,), dtype=np.float32),
            }

        ax, ay = agent.pos
        padded_ax = ax + radius
        padded_ay = ay + radius
        obstacles = self._padded_obstacle_grid[
            padded_ay - radius : padded_ay + radius + 1,
            padded_ax - radius : padded_ax + radius + 1,
        ]

        all_agent_positions.remove(agent.pos)
        all_agent_goal_positions.remove(agent.goal_pos)

        agent_view_cells = self.get_agent_view_cells(agent=agent)
        for cell in agent_view_cells & all_agent_positions:
            view_pos = self._view_position(agent, cell)
            view[1, view_pos[0], view_pos[1]] = 1

        for cell in agent_view_cells & all_agent_goal_positions:
            view_pos = self._view_position(agent, cell)
            view[2, view_pos[0], view_pos[1]] = 1

        self._add_if_in_view(agent, agent.goal_pos, view[3])

        goal_vector = self._goal_vector(agent)

        all_agent_positions.add(agent.pos)
        all_agent_goal_positions.add(agent.goal_pos)

        view[0] = obstacles

        return {
            "view": view,
            "goal_vector": goal_vector,
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

    def render(self, paused=False):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size * 2, self.window_size), pygame.RESIZABLE
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

        current_w, current_h = self.window.get_size()
        canvas = pygame.Surface((current_w, current_h))
        canvas.fill((255, 255, 255))

        scale_x = current_w // self.grid_width
        scale_y = current_h // self.grid_height
        dynamic_cell_size = min(scale_x, scale_y)

        grid_total_width = self.grid_width * dynamic_cell_size
        grid_total_height = self.grid_height * dynamic_cell_size

        # Obliczamy marginesy, aby wyśrodkować siatkę
        offset_x = (current_w - grid_total_width) // 4
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
        for depot in self._get_unique_depots():
            d_rect = pygame.Rect(
                offset_x + depot.pos[0] * dynamic_cell_size,
                offset_y + depot.pos[1] * dynamic_cell_size,
                dynamic_cell_size,
                dynamic_cell_size,
            )
            pygame.draw.rect(canvas, (0, 0, 255), d_rect)

        # 3. Rysowanie Robotów i ich Celów
        text_color = (0, 0, 0)

        # Drawing packages and depots
        for agent in self.agents:
            agent_id_str = f"ID:{agent.id}"
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

        mouse_x, mouse_y = pygame.mouse.get_pos()
        hovered_agent = None

        # Drawing robots
        for agent in self.agents:
            pos_x = offset_x + agent.pos[0] * dynamic_cell_size
            pos_y = offset_y + agent.pos[1] * dynamic_cell_size

            robot_rect = pygame.Rect(
                pos_x,
                pos_y,
                dynamic_cell_size,
                dynamic_cell_size,
            )

            if robot_rect.collidepoint(mouse_x, mouse_y):
                hovered_agent = agent

            robot_scaled = pygame.transform.scale(
                self.robot_img, (dynamic_cell_size, dynamic_cell_size)
            )

            r = ((agent.id + 1) * 50) % 256
            g = ((agent.id + 1) * 80) % 256
            b = ((agent.id + 1) * 110) % 256
            robot_scaled.fill((r, g, b, 255), special_flags=pygame.BLEND_RGBA_MULT)

            canvas.blit(robot_scaled, (pos_x, pos_y))

            agent_id_str = f"ID:{agent.id}"
            agent_text = self.font.render(agent_id_str, True, (0, 0, 0))
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
            r = ((hovered_agent.id + 1) * 50) % 256
            g = ((hovered_agent.id + 1) * 80) % 256
            b = ((hovered_agent.id + 1) * 110) % 256
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
