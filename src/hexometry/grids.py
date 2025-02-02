import heapq

from typing import Callable, Iterator, TypeVar, override

from .coordinates import Coord, Route, get_directed_neighbours


HexValue = TypeVar('HexValue')


class Grid(dict[Coord, HexValue]):
    """Generic Hex Grid of some values."""

    def __init__(self, default: HexValue | Callable[[Coord], HexValue]):
        self.default: HexValue | None = None if callable(default) else default
        self.get_default: Callable | None = default if callable(default) else None

    def normalize(self, hex: Coord, value: HexValue) -> HexValue:
        return value

    def __setitem__(self, key: Coord, value: HexValue) -> None:
        super().__setitem__(key, self.normalize(key, value))

    def __getitem__(self, key: Coord) -> HexValue | None:
        if key in self:
            return super().__getitem__(key)
        elif self.get_default is not None:
            return self.get_default(key)

        return self.default

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}({super().__repr__()})>'

    def hexes(self) -> Iterator[Coord]:
        yield from self.keys()


class BlockageGrid(Grid[float]):
    """Hex grid of float blockage values. Useful for calculating Route cost.
    Blockage values are penalties from 0 to 1
    where 1 means hex at this coordinates is blocked for traversing
    """

    MIN_VALUE = 0.0
    MAX_VALUE = 1.0

    def __init__(self, default_blockage_level: float = 0.0):
        super().__init__(default=default_blockage_level)

    @override
    def normalize(self, hex: Coord, value: float):
        if value < self.MIN_VALUE:
            return self.MIN_VALUE
        if value > self.MAX_VALUE:
            return self.MAX_VALUE

        return value


def dijkstra(start: Coord, end: Coord, penalties: BlockageGrid | None = None, step_penalty=0.00001) -> Route:
    """
    Returns a route from `start` coordinates to `end`, respecting `penalties` grid if provided.
    If there is no route (penalties grid blocks any route) will return empty route [].
    `step_penalty` — minimal penalty to each step to track
        how far from start we get and look for the shortest way.
        Could be useful to balance between minimizing distance or sum of penalties on it.
    """
    if step_penalty <= 0:
        raise ValueError('step_penalty should be positive float number')  # to avoid infinite loops

    if penalties is None:
            # no penalties grid provided, assuming all hexes field is available
            # fallback to cheapest default route calculation
            return start >> end

    queue = [(0, start)]
    distances: dict[Coord, float] = {start: 0.0}
    previous: dict[Coord, Coord] = {}
    directions: dict[Coord, Direction] = {}

    while queue:
        current_distance, current_hex = heapq.heappop(queue)

        if current_hex == end:
            break

        for direction, neighbour in get_directed_neighbours(current_hex):
            if penalties[neighbour] >= BlockageGrid.MAX_VALUE:  # hex is blocked
                continue

            distance_to_neighbour = current_distance + step_penalty + penalties[neighbour]

            if neighbour not in distances or distance_to_neighbour < distances[neighbour]:
                distances[neighbour] = distance_to_neighbour  # best value
                directions[neighbour] = direction  # how we get there
                previous[neighbour] = current_hex  # where we made last step from
                heapq.heappush(queue, (distance_to_neighbour, neighbour))

    # Reconstruct the path
    path = Route()
    while end in previous:
        path.append(directions[end])
        end = previous[end]

    path.reverse()
    return path
