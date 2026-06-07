from collections import deque
from itertools import islice

# if TYPE_CHECKING:
from src.models.depot import Depot
from src.models.task import Task
from src.utils.enums import TaskType


class DeliveryRobot:
    pos: tuple[int, int]
    task: Task
    depot: Depot

    id: int
    was_blocked: bool

    finish_times: dict[TaskType, int]

    step_count: int

    goal_pos: tuple[int, int] | None
    task_type: TaskType | None

    next_pos: tuple[int, int] | None

    pos_history: deque[tuple[int, int]]

    idle_time: int

    def __init__(
        self,
        position: tuple[int, int],
        task: Task,
        depot: Depot,
        id: int,
        finish_times: dict[TaskType, int] | None = None,
    ):
        self.pos = position
        self.task = task
        self.depot = depot
        self.id = id
        self.was_blocked = False
        self.finish_times = finish_times or {
            TaskType.PICKUP: 1,
            TaskType.LEAVE: 1,
        }
        self.step_count: int = 0

        self.goal_pos, self.task_type = self.task.pop_next()
        self.next_pos = None
        self.pos_history = deque()
        self.idle_time = 0

    def step(self) -> bool:
        """
        Returns True if robot should be removed
        """
        # update step count
        self.step_count += 1

        # wait if idle
        if self.idle_time > 0:
            self.idle_time = max(0, self.idle_time - 1)
            return False

        # leave if on depot
        if self.task_type == TaskType.LEAVE and self.pos == self.depot.pos:
            self.depot.stored_agents.append(self)
            return True

        # move
        self.move()

        # finish task and set idle time
        if self.pos == self.goal_pos:
            self.finish_goal()
            self._next_task()

        return False

    def move(self):
        self.pos_history.append(self.pos)
        if self.next_pos:
            self.pos = self.next_pos
            self.next_pos = None

    def finish_goal(self):
        if self.task_type == TaskType.LEAVE:
            self.idle_time = self.finish_times[self.task_type] - 1
        else:
            self.idle_time = self.finish_times[self.task_type]

    def is_stuck(self) -> bool:
        stuck_time = 5
        if len(self.pos_history) < stuck_time:
            return False
        recent_positions = islice(reversed(self.pos_history), stuck_time)
        unique_positions = set(recent_positions)
        return len(unique_positions) <= 2

    def reward(self, next_pos: tuple[int, int], empty_cells: set[tuple[int, int]]):
        reward = -1
        if next_pos == self.pos:
            return reward
        if next_pos not in empty_cells:
            return -20
        if next_pos == self.goal_pos:
            return 100

        old_dist = self._goal_distance(self.pos)
        new_dist = self._goal_distance(next_pos)
        dist_reward = old_dist - new_dist

        if dist_reward < 0:
            dist_reward *= 5
        else:
            dist_reward *= 2.5
        reward += dist_reward

        return reward

    def set_next_pos(self, pos: tuple[int, int]):
        if self.idle_time > 0:
            return
        self.next_pos = pos

    def set_next_goal_pos(self, goal_pos: tuple[int, int]):
        self.goal_pos = goal_pos

    def get_occupied_cells(self) -> list[tuple[int, int]]:
        occupied_cells = [self.pos]
        if self.next_pos is not None:
            occupied_cells.append(self.next_pos)
        return occupied_cells

    def is_done(self) -> bool:
        return not self.is_idle() and self.goal_pos is None and self.task.goals == []

    def is_idle(self) -> bool:
        return self.idle_time > 0

    def _next_task(self) -> None:
        if self.task.is_completed():
            # go to depot
            self.goal_pos, self.task_type = self.depot.pos, TaskType.LEAVE
            return
        self.goal_pos, self.task_type = self.task.pop_next()

    def _goal_distance(self, pos: tuple[int, int]) -> int:
        if self.goal_pos is None:
            raise ValueError("Agent has no goal")
        elif pos is None:
            raise ValueError("Agent has no position")
        return abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

    def __eq__(self, other: "DeliveryRobot"):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"DeliveryRobot(pos={self.pos}, goal={self.goal_pos}, id={self.id})"

    def __str__(self):
        return f"DeliveryRobot(pos={self.pos}, goal={self.goal_pos}, id={self.id})"
