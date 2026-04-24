from src.utils.enums import Task


class DeliveryRobot:
    def __init__(
        self,
        position: tuple[int, int],
        goal: tuple[int, int] | None = None,
        id: int = -1,
        task: Task = Task.PICKUP,
    ):
        self.pos = position
        self.goal = goal
        self.next_pos = None
        self.id = id
        self.task = task
        self.idle_time = 0

    def move(self):
        if self.idle_time > 0:
            self.idle_time = max(0, self.idle_time - 1)
            return

        if self.next_pos:
            self.pos = self.next_pos
            self.next_pos = None

        if self.pos == self.goal:
            self.idle_time = 1
            self.goal = None

    def set_next_pos(self, pos: tuple[int, int]):
        self.next_pos = pos

    def set_next_goal(self, goal: tuple[int, int]):
        self.goal = goal

    def get_occupied_cells(self) -> list[tuple[int, int]]:
        occupied_cells = [self.pos]
        if self.next_pos is not None:
            occupied_cells.append(self.next_pos)
        return occupied_cells

    def __eq__(self, other: "DeliveryRobot"):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)

    def __repr__(self):
        return f"DeliveryRobot(pos={self.pos}, goal={self.goal})"

    def __str__(self):
        return f"DeliveryRobot(pos={self.pos}, goal={self.goal})"
