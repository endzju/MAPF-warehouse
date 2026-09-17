from collections import deque

import numpy as np

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
    action_times: dict[TaskType, int]
    step_count: int
    goal_pos: tuple[int, int] | None
    task_type: TaskType | None
    next_pos: tuple[int, int] | None
    pos_history: deque[tuple[int, int]]
    busy_time: int
    stuck_time: int
    stuck_pos_history: deque[tuple[int, int]]
    cached_view: np.ndarray
    cached_additional_input: np.ndarray

    def __init__(
        self,
        position: tuple[int, int],
        task: Task,
        in_depot: Depot,
        out_depot: Depot,
        id: int,
        cached_view,
        cached_additional_input,
        action_times: dict[TaskType, int] | None = None,
        busy_time=0,
        stuck_time=0,
    ):
        self.pos = position
        self.task = task
        self.in_depot = in_depot
        self.out_depot = out_depot
        self.id = id
        self.was_blocked = False
        self.action_times = action_times or {
            TaskType.PICKUP: 1,
            TaskType.LEAVE: 1,
        }
        self.step_count: int = 0
        self.move_count: int = 0

        if self.task.is_completed():
            raise ValueError("Task must not be completed")
        self.goal_pos, self.task_type = self.task.pop_next()
        self.next_pos = None
        self.pos_history = deque()
        self.busy_time = busy_time
        self.stuck_time = stuck_time
        self.stuck_pos_history = deque(maxlen=self.stuck_time)
        self.should_exit = False
        self.allow_next_observation = False
        self.cached_view = cached_view
        self.cached_additional_input = cached_additional_input

    def step(self):
        """
        Returns True if finished goal
        """
        self.step_count += 1
        finished_goal = False

        # wait if idle
        if self.busy_time > 0:
            self.busy_time = max(0, self.busy_time - 1)
            self.pos_history.append(self.pos)
        else:
            self.move_count += 1

            # move
            self.move()

            # finish task and set idle time
            if self.pos == self.goal_pos:
                finished_goal = True
                self.finish_goal()
                self._next_task()

        return finished_goal

    def move(self):
        if self.next_pos is None:
            self.next_pos = self.pos
        self.pos_history.append(self.pos)
        self.stuck_pos_history.append(self.pos)
        if self.next_pos != self.pos:
            self.busy_time = self.action_times[TaskType.MOVE] - 1
        if self.next_pos:
            self.pos = self.next_pos
            self.next_pos = None

    def finish_goal(self):
        self.busy_time = self.action_times[self.task_type]
        if self.task_type == TaskType.LEAVE:
            self.in_depot.finished_tasks.append(self.task)
            self.should_exit = True
        self.allow_next_observation = True

    def is_stuck(self) -> bool:
        if self.stuck_time == 0:
            return False
        if len(self.stuck_pos_history) < self.stuck_time:
            return False
        unique_positions = set(self.stuck_pos_history)
        return len(unique_positions) <= 2

    def set_next_pos(self, pos: tuple[int, int]):
        if self.busy_time > 0:
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
        return (
            not self.is_busy()
            and self.goal_pos == self.out_depot.pos
            and self.task.is_completed()
        )

    def is_busy(self) -> bool:
        return self.busy_time > 0

    def is_entering_grid(self) -> bool:
        return self.step_count <= self.action_times[TaskType.ENTER]

    def is_leaving_grid(self) -> bool:
        return (
            self.pos_history and self.pos == self.pos_history[-1] == self.out_depot.pos
        )

    def _get_exit_wait_length(self) -> int:
        count = 0
        for idx in range(-1, -self.action_times[TaskType.LEAVE] - 1, -1):
            if self.pos_history[idx] == self.out_depot.pos:
                count += 1
        return count

    def should_return_to_depot(self) -> bool:
        return self.should_exit and not self.is_busy()

    def should_generate_observation(self) -> bool:
        return not self.is_busy()

    def _next_task(self) -> None:
        if self.should_exit:
            return
        if self.task.is_completed():
            self.goal_pos, self.task_type = self.out_depot.pos, TaskType.LEAVE
            return
        self.goal_pos, self.task_type = self.task.pop_next()

    def _goal_distance(self, pos: tuple[int, int]) -> int:
        if self.goal_pos is None:
            raise ValueError("Agent has no goal")
        elif pos is None:
            raise ValueError("Agent has no position")
        return abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

    def reset(self):
        self.busy_time = 0
        self.step_count = 0
        self.move_count = 0
        self.was_blocked = False
        self.pos_history = deque()
        self.stuck_pos_history = deque(maxlen=self.stuck_time)
        self.next_pos = None
        self.task_type = None
        self.goal_pos = None
        self.should_exit = False
        self.allow_next_observation = False

    def __eq__(self, other: "DeliveryRobot"):
        return isinstance(other, DeliveryRobot) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"DeliveryRobot(pos={self.pos}, goal={self.goal_pos}, id={self.id})"

    def __str__(self):
        return f"DeliveryRobot(pos={self.pos}, goal={self.goal_pos}, id={self.id})"
