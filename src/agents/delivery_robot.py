from typing import TYPE_CHECKING

from src.agents.task import Task
from src.utils.enums import TaskType

if TYPE_CHECKING:
    from src.agents.depot import Depot


class DeliveryRobot:
    def __init__(
        self,
        position: tuple[int, int],
        task: Task,
        depot: "Depot",
        id: int = -1,
    ):
        self.pos = position
        self.task = task
        self.depot: Depot = depot
        self.id = id

        self.goal_pos = task.goals[0]
        self.task_type = task.goalTypes[0]
        self.next_pos = None
        self.idle_time = 0
        self._next_task()

    def step(self) -> bool:
        """
        Returns True if robot should be removed
        """
        if self.idle_time > 0:
            self.idle_time = max(0, self.idle_time - 1)
            return False

        if self.next_pos:
            self.pos = self.next_pos
            self.next_pos = None

        if self.pos == self.goal_pos:
            self.idle_time = 1
            self._next_task()
            return False

        if self.task_type == TaskType.LEAVE and self.pos == self.depot.pos:
            self.depot.stored_agents.append(self)
            return True

        return False

    def reward(self, next_pos: tuple[int, int], empty_cells: set[tuple[int, int]]):
        reward = -0.1
        if next_pos == self.pos:
            return reward
        if next_pos not in empty_cells:
            return -10
        if next_pos == self.goal_pos:
            return 100

        old_dist = self._goal_distance(self.pos)
        new_dist = self._goal_distance(next_pos)
        reward += old_dist - new_dist

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
        if len(self.task.goals) == 0:
            self.goal_pos, self.task_type = None, TaskType.LEAVE
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
