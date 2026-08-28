from collections import deque
from itertools import islice

# if TYPE_CHECKING:
from src.models.depot import Depot
from src.models.task import Task
from src.utils.distance import manhattan_distance
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

    def __init__(
        self,
        position: tuple[int, int],
        task: Task,
        in_depot: Depot,
        out_depot: Depot,
        id: int,
        action_times: dict[TaskType, int] | None = None,
        busy_time=0,
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
        self.goal_pos, self.task_type = None, None
        self.next_pos = None
        self.pos_history = deque()
        self.busy_time = busy_time
        self.should_exit = False

    def step(self):
        """
        Returns True if robot should be removed
        """
        # print("debug", self.id, self.pos, self.task_type, self.goal_pos, self.busy_time)
        if self.goal_pos is None and not self.task.is_completed():
            self.goal_pos, self.task_type = self.task.pop_next()

        # update step count
        self.step_count += 1

        # wait if idle
        if self.busy_time > 0:
            self.busy_time = max(0, self.busy_time - 1)
            self.pos_history.append(self.pos)
            return

        # leave if on depot
        if self.task_type == TaskType.LEAVE and self.pos == self.out_depot.pos:
            # print(
            #     "debug1",
            #     self._get_exit_wait_length(),
            #     self.action_times[TaskType.LEAVE],
            #     self.pos,
            # )
            if self._get_exit_wait_length() == self.action_times[TaskType.LEAVE] - 1:
                self.pos_history.append(self.pos)
                self.in_depot.finished_tasks.append(self.task)
                # print("SETTING TASK AS DONE, robotID:", self.id, self.pos)
                self.should_exit = True
            else:
                self.pos_history.append(self.pos)
            return

        # move
        self.move()

        # finish task and set idle time
        if self.pos == self.goal_pos:
            self.finish_goal()
            self._next_task()

        return

    def move(self):
        if self.next_pos is None:
            self.next_pos = self.pos
        self.pos_history.append(self.pos)
        move_time = 1
        if self.next_pos != self.pos:
            self.busy_time = self.action_times[TaskType.MOVE] - 1
        if self.next_pos:
            self.pos = self.next_pos
            self.next_pos = None

    def finish_goal(self):
        if self.task_type == TaskType.LEAVE:
            return
        self.busy_time = self.action_times[self.task_type]

    def is_stuck(self) -> bool:
        if (
            self.pos == self.goal_pos
            or manhattan_distance(self.pos, self.in_depot.pos) < 3
            or manhattan_distance(self.pos, self.out_depot.pos) < 3
        ):
            return False
        stuck_time = 5
        if len(self.pos_history) < stuck_time:
            return False
        recent_positions = islice(reversed(self.pos_history), stuck_time)
        unique_positions = set(recent_positions)
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
            not self.is_idle()
            and self.goal_pos == self.out_depot.pos
            and self.task.is_completed()
        )

    def is_busy(self) -> bool:
        return self.busy_time > 0

    def is_entering_grid(self) -> bool:
        return len(self.pos_history) <= self.action_times[TaskType.ENTER]

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
        return self.should_exit

    def _next_task(self) -> None:
        if self.task.is_completed():
            # go to depot
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
        self.next_pos = None
        self.task_type = None
        self.goal_pos = None
        self.should_exit = False

    def __eq__(self, other: "DeliveryRobot"):
        return isinstance(other, DeliveryRobot) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"DeliveryRobot(pos={self.pos}, goal={self.goal_pos}, id={self.id})"

    def __str__(self):
        return f"DeliveryRobot(pos={self.pos}, goal={self.goal_pos}, id={self.id})"
