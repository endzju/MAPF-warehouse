class Depot:
    def __init__(self, pos: tuple[int, int] = (0, 0)):
        self.pos = pos
        self.stored_agents = []
        self.task_history = []

    def __hash__(self):
        return hash(self.pos)

    def __eq__(self, other):
        return isinstance(other, Depot) and self.pos == other.pos
