import heapq

from typing import Callable, Iterator, TypeVar

from .coordinates import Coord, Direction, Route, get_directed_neighbours, get_distance


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
    """Hex grid of float blockage values.

    Useful for calculating Route cost.
    Blockage values are penalties from 0.0 to 1.0
    where 1.0 means hex at this coordinates is blocked for traversing.
    """

    MIN_VALUE = 0.0
    MAX_VALUE = 1.0

    def __init__(self, radius: int, default_blockage_level: float = MIN_VALUE):
        if radius <= 0:
            raise ValueError('BlockageGrid radius should be positive integer value')
        self.grid_radius = radius
        self.default_blockage_level = default_blockage_level
        super().__init__(default=self._get_blocked_areas)

    def normalize(self, hex: Coord, value: float):
        if value < self.MIN_VALUE:
            return self.MIN_VALUE
        if value > self.MAX_VALUE:
            return self.MAX_VALUE

        return value

    def _get_blocked_areas(self, hex: Coord | tuple[int, int]) -> float:
        """For hexes out of the grid_radius scope return maximum blockage value.
        This needed to limit traversing route finding algorythms
        to not go far away from center into empty fields, looking for less penalties.
        """
        if isinstance(hex, tuple):
            hex = Coord(*hex)

        z_value = 0 - hex.x - hex.y

        if abs(hex.x) >= self.grid_radius:
            return self.MAX_VALUE
        if abs(hex.y) >= self.grid_radius:
            return self.MAX_VALUE
        if abs(z_value) >= self.grid_radius:
            return self.MAX_VALUE
        return self.default_blockage_level

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}[R={self.grid_radius}, Size={len(self)}]({dict(self)})>'


def dijkstra(start: Coord, end: Coord, penalties: BlockageGrid | None = None, step_penalty=0.001) -> Route:
    """
    Returns a route from `start` coordinates to `end`, respecting `penalties` grid if provided.
    If there is no route (penalties grid blocks any route) — will return empty route [].
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

    queue = [(0, start)]  # at the starting hex we have 0 penalties
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


def a_star(start: Coord, end: Coord, penalties: BlockageGrid | None = None, step_penalty=0.00001) -> Route:
    """
    A* pathfinding algorithm for hexagonal grids.
    Returns a route from `start` coordinates to `end`, respecting `penalties` grid if provided.
    If there is no route (penalties grid blocks any route) will return empty route [].

    `step_penalty` — minimal penalty to each step to track
        how far from start we get and look for the shortest way.
        Could be useful to balance between minimizing distance or sum of penalties on it.

    Unlike Dijkstra, A* uses a heuristic to prioritize paths that seem to lead toward the destination.
    This makes it more efficient for finding routes between distant points.
    """
    if step_penalty <= 0:
        raise ValueError('step_penalty should be positive float number')  # to avoid infinite loops

    if penalties is None:
        # no penalties grid provided, assuming all hexes field is available
        # fallback to cheapest default route calculation
        return start >> end

    # Heuristic function: estimate of distance from current to end
    def heuristic(hex: Coord) -> float:
        return get_distance(hex, end) * step_penalty

    # The set of discovered nodes that need to be evaluated
    open_set = [(heuristic(start), 0, start)]  # (f_score, g_score, hex)
    g_score: dict[Coord, float] = {start: 0.0}  # cost of the cheapest path from start
    f_score: dict[Coord, float] = {start: g_score[start] + heuristic(start)}

    previous: dict[Coord, Coord] = {}
    directions: dict[Coord, Direction] = {}
    while open_set:
        _, current_g_score, current_hex = heapq.heappop(open_set)  # pop lowest f_score

        # If we've reached the end, reconstruct and return the path
        if current_hex == end:
            path = Route()
            current = end
            while current in previous:
                path.append(directions[current])
                current = previous[current]
            path.reverse()
            return path

        for direction, neighbor in get_directed_neighbours(current_hex):
            if penalties[neighbor] >= BlockageGrid.MAX_VALUE:  # hex is blocked
                continue

            tentative_g_score = current_g_score + step_penalty + penalties[neighbor]

            # If this path to neighbor is better than any previous one, record it
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                previous[neighbor] = current_hex
                directions[neighbor] = direction
                g_score[neighbor] = tentative_g_score
                new_f_score = tentative_g_score + heuristic(neighbor)
                f_score[neighbor] = new_f_score

                # Add to open set if not already there or update its priority
                heapq.heappush(open_set, (new_f_score, tentative_g_score, neighbor))

    return Route()  # if there is no path, return an empty Route
