from src.utils.enums import TaskType


class Task:
    def __init__(
        self,
        goal_positions: list[tuple[int, int]],
        goal_types: list[TaskType],
        id: int = -1,
    ):
        self.goal_positions = goal_positions
        self.goal_types = goal_types
        self.task_index = 0
        if self.goal_types is None:
            raise ValueError("Task must have goal_types")
        self.id = id

    def pop_next(self) -> tuple[tuple[int, int], TaskType]:
        if self.task_index == len(self.goal_positions):
            raise IndexError(
                f"No more tasks, task_id{self.id}, task_index{self.task_index}"
            )
        goal = self.goal_positions[self.task_index]
        goalType = self.goal_types[self.task_index]
        self.task_index += 1
        return goal, goalType

    def is_completed(self) -> bool:
        return self.task_index == len(self.goal_positions)

    def get_delivered_goals(self) -> list[tuple[int, int]]:
        return self.goal_positions[: self.task_index - 1]

    def reset(self):
        self.task_index = 0

    def __len__(self):
        return len(self.goal_positions)
