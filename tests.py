import functools
import operator
import random
import types

import pytest

from hexometry import (
    Coord,
    Direction,
    Route,
    get_route,
    get_directed_neighbours,
    hex_to_decart,
    hex_to_decart_corners,
    decart_to_hex,
    get_neighbour,
    get_neighbours,
    get_distance,
)


def test_direction_sequence():
    assert list(Direction) == [
        Direction.NE,
        Direction.E,
        Direction.SE,
        Direction.SW,
        Direction.W,
        Direction.NW,
    ]


def test_direction_turns():
    assert -Direction.NE == Direction.NW
    assert Direction.NE == +Direction.NW
    assert Direction.E == ---Direction.W
    assert +-+-+-+-Direction.W == Direction.W
    assert ++++----Direction.SE == Direction.SE
    assert ~Direction.NE == ---Direction.NE

    assert ~Direction.E == Direction.W
    assert ~Direction.W == Direction.E

    assert ~Direction.SE == Direction.NW
    assert Direction.NW == ~Direction.SE


def test_direction_multiplication():
    for d in Direction:
        assert d * 3 == [d] * 3 == [d, d, d] == Route([d] * 3)


def test_direction_repr():
    for d in Direction:
        assert repr(d) == str(d.value)


# list of neighbour coordinates by distance
neighbours_by_coordinates = {
    (0, 0): {
        0: [],
        1: [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)],
        2: [
            (0, 2), (1, 1), (2, 0), (2, -1), (2, -2), (1, -2),
            (0, -2), (-1, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        ],
    },
    (12, 34): {
        0: [],
        1: [(12, 35), (13, 34), (13, 33), (12, 33), (11, 34), (11, 35)],
        2: [
            (12, 36), (13, 35), (14, 34), (14, 33), (14, 32), (13, 32),
            (12, 32), (11, 33), (10, 34), (10, 35), (10, 36), (11, 36),
        ],
    },
    (-10, 30): {
        0: [],
        1: [(-10, 31), (-9, 30), (-9, 29), (-10, 29), (-11, 30), (-11, 31)],
        2: [
            (-10, 32), (-9, 31), (-8, 30), (-8, 29), (-8, 28), (-9, 28),
            (-10, 28), (-11, 29), (-12, 30), (-12, 31), (-12, 32), (-11, 32),
        ],
    },
}


@pytest.mark.parametrize('coordinates, nearest_neighbours', [(c, n[1]) for c, n in neighbours_by_coordinates.items()])
def test_get_neighbor_by_direction(coordinates, nearest_neighbours):
    """Test get_neighbour function with different types of arguments."""
    c = Coord(*coordinates)
    for direction, expected_neighbour in zip(Direction, nearest_neighbours):
        assert c + direction == Coord(*expected_neighbour)
        assert get_neighbour(c, direction) == Coord(*expected_neighbour)
        assert get_neighbour(c, direction.value) == Coord(*expected_neighbour)
        assert get_neighbour(coordinates, direction) == Coord(*expected_neighbour)
        assert get_neighbour(coordinates, direction.value) == Coord(*expected_neighbour)
    

@pytest.mark.parametrize('coordinates, neighbours_by_distance', neighbours_by_coordinates.items())
def test_get_neighbors_by_distance(coordinates, neighbours_by_distance):    
    c = Coord(*coordinates)

    for distance, expected_neighbours in neighbours_by_distance.items():
        neighbours_by_coord = get_neighbours(c, distance=distance)
        neighbours_by_xy = get_neighbours(coordinates, distance=distance)

        assert isinstance(neighbours_by_coord, types.GeneratorType)
        assert isinstance(neighbours_by_xy, types.GeneratorType)

        assert list(neighbours_by_coord) == [Coord(*n) for n in expected_neighbours]
        assert list(neighbours_by_xy) == [Coord(*n) for n in expected_neighbours]


@pytest.mark.parametrize('coordinates, neighbours_by_distance', neighbours_by_coordinates.items())
def test_get_neighbors_within_distance(coordinates, neighbours_by_distance):
    c = Coord(*coordinates)
    expected = []

    for distance, expected_neighbours in neighbours_by_distance.items():
        expected.extend(expected_neighbours)
        neighbours = get_neighbours(c, distance=distance, within=True)
        assert list(neighbours) == [Coord(*n) for n in expected]


def test_coord_repr():
    c = Coord(-1, 2)
    assert repr(c) == '[-1:2]'


def test_get_neighbor_for_big_distance_works():
    assert len(list(get_neighbours((123, 234), distance=3000))) == 18000
    assert len(list(get_neighbours((123, 234), distance=300, within=True))) == 270900


def test_get_directed_neighbours():
    c = Coord(12, -34)

    neighbours = get_directed_neighbours(c)
    assert isinstance(neighbours, types.GeneratorType)

    for d, n in neighbours:
        assert c + d == n


def test_empty_route():
    c = Coord(123, 456)
    assert get_route(c, c) == []
    assert c >> c == []
    assert c << c == []
    assert Coord(123, 456) >> c == []


def test_get_route():
    c1 = Coord(-3, 5)
    c2 = Coord(2, 7)

    route = get_route(c1, c2)
    assert route == [
        *[Direction.E] * 5,
        *[Direction.NE] * 2,
    ]

    assert c1 >> c2 == route
    assert c1 >> c2 == c2 << c1


def test_traverse_routes():
    c1 = Coord(random.randint(-100, 100), random.randint(-100, 100))
    c2 = Coord(random.randint(-100, 100), random.randint(-100, 100))

    assert c1 * (c1 >> c2) == c2
    assert c2 * (c1 << c2) == c1

    assert c1 * (c1 >> c2) * (c2 >> c1) == c1
    assert c1 * (Direction.NE * 3) * (Direction.SE * 3) * (Direction.W * 3) == c1

    assert c1 + '↖' + '↗' + '→' + '↘' + '↙' + '←' == c1

    assert functools.reduce(operator.add, c1 >> c2, c1) == c2


def test_get_rote_shuffled():
    c1 = Coord(-3, 5)
    c2 = Coord(2, -7)

    for _ in range(100):
        route = get_route(c1, c2, shuffle=True)
        assert c1 * route == c2


def test_get_distance():
    c1 = Coord(-4, -3)
    c2 = Coord(2, 8)

    assert get_distance(c1, c2) == get_distance(c2, c1)
    assert get_distance(c1, c2) == len(c1 >> c2)
    assert get_distance(c1, c2) == c1 - c2
    assert get_distance(c1, c2) == 17
    assert c1 - c2 == c2 - c1



coordinates_test_cases = [
    # hex_xy, scale_factor, dec_xy
    ((0, 0), 1, (0.0, 0.0)),
    ((0, 0), 2, (0.0, 0.0)),
    ((-4, 3), 1, (-6.0, 1.7321)),
    ((-4, 3), -2, (12.0, -3.4641)),
    ((8, 11), -3.1415, (-37.698, -81.6186)),
]


@pytest.mark.parametrize('hex_xy, scale_factor, dec_xy', coordinates_test_cases)
def test_hex_to_decart(hex_xy, scale_factor, dec_xy):
    c = Coord(*hex_xy)

    assert hex_to_decart(c, scale_factor=scale_factor) == dec_xy
    assert hex_to_decart(hex_xy, scale_factor=scale_factor) == dec_xy
    assert ~c == hex_to_decart(c, scale_factor=1)
    assert hex_to_decart(c, scale_factor=-scale_factor) == (-dec_xy[0], -dec_xy[1])


def test_scale_factor_zero():
    c = Coord(random.randint(1, 10), random.randint(1, 10))
    assert hex_to_decart(c, scale_factor=0) == (0, 0)
    assert decart_to_hex(random.random() * 10, random.random() * 10, scale_factor=0) == (0, 0)


@pytest.mark.parametrize('hex_xy, scale_factor, dec_xy', coordinates_test_cases)
def test_decart_to_hex(hex_xy, scale_factor, dec_xy):
    assert decart_to_hex(*dec_xy, scale_factor=scale_factor) == Coord(*hex_xy)
    assert decart_to_hex(*hex_to_decart(hex_xy, scale_factor=scale_factor), scale_factor=scale_factor) == hex_xy



def test_coord_from_decart():
    assert Coord.from_decart(*~Coord(23, 45)) == Coord(23, 45)

    assert Coord.from_decart(184.5, -683.294) == Coord(123, -456)
    assert Coord.from_decart(184.9, -683.3) == Coord(123, -456)
    assert Coord.from_decart(185, -683) == Coord(123, -456)

    assert Coord.from_decart(-579.60675, 2146.568238, scale_factor=3.14) == Coord(-123, 456)


hex_corners_coordinates_test_cases = [
    # hex_xy, scale_factor, expected_xy
    ((0, 0), 1, [(0.5, 0.866), (1.0, 0.0), (0.5, -0.866), (-0.5, -0.866), (-1.0, 0.0), (-0.5, 0.866)]),
    ((0, 0), 2, [(1.0, 1.7321), (2.0, 0.0), (1.0, -1.7321), (-1.0, -1.7321), (-2.0, 0.0), (-1.0, 1.7321)]),
    (
        (-4, 3),
        1,
        [
            (-5.5, 2.5981),
            (-5.0, 1.7321),
            (-5.5, 0.8661),
            (-6.5, 0.8661),
            (-7.0, 1.7321),
            (-6.5, 2.5981)
        ]
    ),
    (
        (-4, 3),
        -3.27,
        [
            (17.985, -8.4957),
            (16.35, -5.6638),
            (17.985, -2.8319),
            (21.255, -2.8319),
            (22.89, -5.6638),
            (21.255, -8.4957),
        ]
    ),
]


@pytest.mark.parametrize('hex_xy, scale_factor, expected_corners_coordinates', hex_corners_coordinates_test_cases)
def test_hex_to_decart_corners(hex_xy, scale_factor, expected_corners_coordinates):
    c = Coord(*hex_xy)
    assert hex_to_decart_corners(c, scale_factor=scale_factor) == expected_corners_coordinates
    assert hex_to_decart_corners(hex_xy, scale_factor=scale_factor) == expected_corners_coordinates
    assert hex_to_decart_corners(c, scale_factor=-scale_factor) == [(-x, -y) for x, y in expected_corners_coordinates]

    if scale_factor == 1:
        assert round(c) == hex_to_decart_corners(c, scale_factor=1)
