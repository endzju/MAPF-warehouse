from src.utils.enums import TaskType


class Task:
    def __init__(
        self,
        goals: list[tuple[int, int]],
        goalTypes: list[TaskType] | None = None,
        id: int = -1,
    ):
        self.goals = goals
        self.goalTypes = goalTypes
        if self.goalTypes is None:
            self.goalTypes = [TaskType.PICKUP] * len(goals - 1) + [TaskType.LEAVE]
        self.goals.reverse()
        self.goalTypes.reverse()
        self.id = id

    def pop_next(self) -> tuple[tuple[int, int], TaskType]:
        if len(self.goals) == 0:
            raise IndexError(f"No more tasks, task_id{self.id}")
        goal = self.goals.pop()
        goalType = self.goalTypes.pop()
        return goal, goalType
