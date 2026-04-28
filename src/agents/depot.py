class Depot:
    def __init__(self, pos: tuple[int, int] = (0, 0)):
        self.pos = pos
        self.stored_agents = []
