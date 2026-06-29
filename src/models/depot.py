from src.models.task import Task


class Depot:
    def __init__(self, pos: tuple[int, int] = (0, 0)):
        self.pos = pos
        # self.stored_agents = []
        # self.task_history = []
        self.tasks = []
        self.finnished_tasks = []
        self.max_robots = 1

    def pop_task(self) -> Task:
        return self.tasks.pop()

    def has_tasks(self) -> bool:
        return len(self.tasks) > 0

    def add_task(self, task: Task):
        self.tasks.append(task)

    def _clear_tasks(self):
        self.tasks = []

    def __hash__(self):
        return hash(self.pos)

    def __eq__(self, other):
        return isinstance(other, Depot) and self.pos == other.pos
