from pyvrp import Model
from pyvrp.stop import MaxRuntime


class pyvrp_solver:
    def __init__(self):
        pass

    def solve(self, coords):
        """Solves the TSP using pyvrp
        Args:
            coords (list[tuple]): List of coordinates

        Returns:
            list[int]: List of coordinates

        """
        coords = [
            (0, 0),
            (10, 5),
            (20, 10),
            (15, 20),
            (5, 15),
        ]

        m = Model()

        m.add_vehicle_type(
            num_available=1,
            capacity=10_000,
        )

        m.add_depot(
            x=coords[0][0],
            y=coords[0][1],
        )

        for x, y in coords[1:]:
            m.add_client(x=x, y=y)

        for frm in m.locations:
            for to in m.locations:
                distance = abs(frm.x - to.x) + abs(frm.y - to.y)
                m.add_edge(
                    frm,
                    to,
                    distance=distance,
                )

        result = m.solve(
            stop=MaxRuntime(1),
            seed=42,
        )

        return list(result.best.routes()[0])
