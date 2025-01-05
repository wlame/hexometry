"""Hexometry module"""

import collections
import enum
import math
import random
import functools

from typing import TypeAlias, Iterator, Callable


__version__ = '0.0.1'


_FLOAT_PRECISION = 4


class Direction(enum.Enum):
    """Hexagonal directions"""

    NE = '↗'
    E = '→'
    SE = '↘'
    SW = '↙'
    W = '←'
    NW = '↖'

    @functools.cached_property
    def _all(self) -> list['Direction']:
        return list(Direction)

    def __repr__(self) -> str:
        return self.value

    def __invert__(self) -> 'Direction':
        """turn 180 degrees"""
        return ---self

    def __neg__(self) -> 'Direction':
        """turn counter-clockwise"""
        index = self._all.index(self)
        return self._all[index - 1]

    def __pos__(self) -> 'Direction':
        """turn clockwise"""
        index = self._all.index(self)
        return self._all[(index + 1) % len(self._all)]

    def __mul__(self, n: int) -> list['Direction']:
        return [self] * n

    def __hash__(self) -> int:
        return hash(self.value)


Route: TypeAlias = list[Direction]
DecartCoord: TypeAlias = tuple[float, float]


class Coord(collections.namedtuple('Coord', ['x', 'y'])):
    """Hexagonal coordinate.
    While `x` and `y` are enough to identify a hexagon,
    `z` coordinate can be calculated as -x-y and is useful in some calculations.
    Refer to this article to get the idea: https://catlikecoding.com/unity/tutorials/hex-map/part-1/
    """

    def __repr__(self) -> str:
        return f'[{self.x}:{self.y}]'

    @property
    def __invert__(self) -> Callable[..., DecartCoord]:
        return lambda: hex_to_decart(self, scale_factor=1)

    @classmethod
    def from_decart(cls, x: float, y: float, scale_factor: float = 1) -> 'Coord':
        return decart_to_hex(x, y, scale_factor=scale_factor)

    def __rshift__(self, other: 'Coord') -> Route:
        return get_route(self, other)

    def __lshift__(self, other: 'Coord') -> Route:
        return get_route(other, self)

    def __mul__(self, route: Route) -> 'Coord':  # type: ignore
        return traverse_route(self, route)

    def __add__(self, direction: Direction | str) -> 'Coord':  # type: ignore
        return get_neighbour(self, direction)

    def __sub__(self, other: 'Coord') -> int:
        return get_distance(self, other)

    def __round__(self, scale_factor: float = 1) -> list[DecartCoord]:
        return hex_to_decart_corners(self, scale_factor=scale_factor)


def get_route(start: Coord, end: Coord, shuffle: bool = False) -> Route:
    """Returns a route from a start coordinate to an end coordinate
    if shuffle is True, directions in a route will be shuffled
    """
    # grad_x: {grad_y: direction} — coordinates gradients for each direction
    gradients = {
        0: {1: Direction.NE, -1: Direction.SW},
        1: {0: Direction.E, -1: Direction.SE},
        -1: {0: Direction.W, 1: Direction.NW},
    }

    dx = end.x - start.x
    dy = end.y - start.y

    route = []
    while dx != 0 or dy != 0:
        grad_x = int(math.copysign(1, dx)) if dx != 0 else 0
        grad_y = int(math.copysign(1, dy)) if dy != 0 else 0

        grad_y = grad_y if grad_y in gradients[grad_x] else next(iter(gradients[grad_x]))

        route.append(gradients[grad_x][grad_y])

        dx -= grad_x
        dy -= grad_y

    if shuffle:
        random.shuffle(route)

    return route


# lambda functions for getting coordinates of a neighbour hex in a given direction
NEIGHBOUR_GETTERS = {
    Direction.NE: lambda x, y: (x, y + 1),  # z-1
    Direction.NE.value: lambda x, y: (x, y + 1),  # z-1
    Direction.E: lambda x, y: (x + 1, y),  # z-1
    Direction.E.value: lambda x, y: (x + 1, y),  # z-1
    Direction.SE: lambda x, y: (x + 1, y - 1),  # z
    Direction.SE.value: lambda x, y: (x + 1, y - 1),  # z
    Direction.SW: lambda x, y: (x, y - 1),  # z+1
    Direction.SW.value: lambda x, y: (x, y - 1),  # z+1
    Direction.W: lambda x, y: (x - 1, y),  # z+1
    Direction.W.value: lambda x, y: (x - 1, y),  # z+1
    Direction.NW: lambda x, y: (x - 1, y + 1),  # z
    Direction.NW.value: lambda x, y: (x - 1, y + 1),  # z
}


def get_neighbour(coord: Coord, direction: Direction | str) -> Coord:
    """Returns the coordinate of the neighbour in the given direction"""
    return Coord(*NEIGHBOUR_GETTERS[direction](*coord))


def get_neighbours(coord: Coord, distance: int = 1, within: bool = False) -> Iterator[Coord]:
    """Generator, yields neighbouring coordinates within a given distance

    If within is True, yields all coordinates within the distance,
    otherwise yields only coordinates exactly at the range distance.
    """
    if distance == 0:
        return

    if distance == 1:
        for direction in Direction:
            yield get_neighbour(coord, direction)
        return

    if within:
        for d in range(distance + 1):
            yield from get_neighbours(coord, d, within=False)
        return

    # make a `distance` steps to the North-East and then traverse that hexagon in a clockwise direction
    hex = traverse_route(coord, [Direction.NE] * distance)
    # to traverse hexagon the the same orientation as in get_neighbour, we start from South-East direction
    directions = list(Direction)[2:] + list(Direction)[:2]
    for direction in directions:
        for _ in range(distance):
            yield hex
            hex = get_neighbour(hex, direction)


def get_directed_neighbours(coord: Coord) -> Iterator[tuple[Direction, Coord]]:
    """Returns a list of neighbouring coordinates, with the direction to each neighbour"""
    yield from zip(Direction, get_neighbours(coord))


def get_distance(a: Coord, b: Coord) -> int:
    """Returns the distance between two hex coordinates as a number of steps in the shortest route
    Adding two coordinates together will always result in a third coordinate with a sum of 0
    Therefore, we can calculate the distance between two coordinates by finding
    the absolute value of the difference between each coordinate's `x`, `y`, and `z` values
    """
    x1, y1 = a
    z1 = -x1 - y1
    x2, y2 = b
    z2 = -x2 - y2
    return max(
        abs(x1 - x2),
        abs(y1 - y2),
        abs(z1 - z2),
    )


def traverse_route(start: Coord, route: Route) -> Coord:
    """Traverses a route from a given starting coordinate, returning the final coordinate"""
    coord = start
    for direction in route:
        coord = get_neighbour(coord, direction)
    return coord


def hex_to_decart(coord: Coord, scale_factor: float) -> DecartCoord:
    """Converts a hex coordinate to a decart coordinates
    assuming the (0, 0) coordinates are matched in hex grid and decart grid
    """
    hex_x, hex_y = coord
    if scale_factor == 0:
        return (0.0, 0.0)

    decart_x = 3 / 2 * hex_x * scale_factor
    decart_y = (3**0.5 / 2 * hex_x + 3**0.5 * hex_y) * scale_factor

    return round(decart_x, _FLOAT_PRECISION), round(decart_y, _FLOAT_PRECISION)


def decart_to_hex(x: float, y: float, scale_factor: float) -> Coord:
    """Converts a decart coordinates to a hex coordinate object"""
    if scale_factor == 0:
        return Coord(0, 0)

    hx = 2 / 3 * x
    hy = -1 / 3 * x + 3**0.5 / 3 * y
    return Coord(round(hx / scale_factor), round(hy / scale_factor))


def hex_to_decart_corners(coord: Coord, scale_factor: float) -> list[DecartCoord]:
    """Converts a hex coordinate to a list of decart coordinates of its corners"""
    x, y = hex_to_decart(coord, scale_factor=scale_factor)  # decart coordinates of the center of the hex
    coordinates = [
        (x + scale_factor / 2, y + 3**0.5 / 2 * scale_factor),
        (x + scale_factor, y),
        (x + scale_factor / 2, y - 3**0.5 / 2 * scale_factor),
        (x - scale_factor / 2, y - 3**0.5 / 2 * scale_factor),
        (x - scale_factor, y),
        (x - scale_factor / 2, y + 3**0.5 / 2 * scale_factor),
    ]
    return [(round(x, _FLOAT_PRECISION), round(y, _FLOAT_PRECISION)) for x, y in coordinates]
