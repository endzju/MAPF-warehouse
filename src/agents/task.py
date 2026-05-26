from src.utils.enums import TaskType


class Task:
    def __init__(
        self,
        goals: list[tuple[int, int]],
        goalTypes: list[TaskType],
        id: int = -1,
    ):
        self.goals = goals
        self.goalTypes = goalTypes
        self.pos_index = 0
        if self.goalTypes is None:
            raise ValueError("Task must have goalTypes")
        self.id = id

    def pop_next(self) -> tuple[tuple[int, int], TaskType]:
        if self.pos_index == len(self.goals):
            raise IndexError(f"No more tasks, task_id{self.id}")
        goal = self.goals[self.pos_index]
        goalType = self.goalTypes[self.pos_index]
        self.pos_index += 1
        return goal, goalType

    def is_completed(self) -> bool:
        return self.pos_index == len(self.goals)

    def get_delivered_goals(self) -> list[tuple[int, int]]:
        return self.goals[: self.pos_index - 1]
