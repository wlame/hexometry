import itertools
import pytest

from hexometry.coordinates import Coord, Direction, Route
from hexometry.grids import Grid, BlockageGrid, dijkstra, a_star


find_path_algorythms = [a_star, dijkstra]


def test_grid_repr():
    grid = Grid(default=None)
    assert repr(grid) == '<Grid({})>'


def test_blockage_grid_repr():
    grid = BlockageGrid(radius=15, default_blockage_level=BlockageGrid.MIN_VALUE)
    assert repr(grid) == '<BlockageGrid[R=15, Size=0]({})>'


def test_grid_initialization_with_default_value():
    grid = Grid(default=0)
    assert grid.default == 0


def test_grid_initialization_with_callable_default():
    def default(coord):
        return coord.x + coord.y

    grid = Grid(default=default)
    assert grid.default is None
    assert grid[Coord(1, 2)] == 3


def test_grid_setitem_normalizes_value():
    class TestGrid(Grid[int]):
        def normalize(self, hex: Coord, value: int):
            return value * 2

    grid = TestGrid(default=0)
    grid[Coord(1, 2)] = 5
    assert grid[Coord(1, 2)] == 10


def test_grid_getitem_returns_stored_value():
    grid = Grid(default=0)
    grid[Coord(1, 2)] = 5
    assert grid[Coord(1, 2)] == 5


def test_grid_getitem_returns_default_value():
    grid = Grid(default=0)
    assert grid[Coord(3, 4)] == 0


def test_grid_getitem_with_callable_default():
    def get_z_coord(coord):
        return 0 - coord.x - coord.y

    grid = Grid(default=get_z_coord)
    assert grid[Coord(1, 2)] == -3


def test_grid_getitem_calls_callable_default():
    def default(coord):
        return coord.x + coord.y

    grid = Grid(default=default)
    assert grid[Coord(1, 2)] == 3


def test_grid_hexes_iterator():
    grid = Grid(default=0)
    grid[Coord(1, 2)] = 5
    grid[Coord(3, 4)] = 10
    hexes = list(grid.hexes())
    assert len(hexes) == 2
    assert Coord(1, 2) in hexes
    assert Coord(3, 4) in hexes


def test_blockagegrid_initialization():
    blockage_grid = BlockageGrid(radius=15, default_blockage_level=0.5)
    assert blockage_grid.grid_radius == 15
    assert blockage_grid.default_blockage_level == 0.5
    assert blockage_grid.get_default == blockage_grid._get_blocked_areas


def test_blockagegrid_negative_radius():
    with pytest.raises(ValueError):
        BlockageGrid(radius=0)

    with pytest.raises(ValueError):
        BlockageGrid(radius=-15)


def test_blockagegrid_normalize_clamps_values():
    blockage_grid = BlockageGrid(15)
    assert blockage_grid.normalize(Coord(1, 2), -0.1) == 0.0
    assert blockage_grid.normalize(Coord(1, 2), 1.5) == 1.0
    assert blockage_grid.normalize(Coord(1, 2), 0.7) == 0.7


def test_blockagegrid_getitem_returns_clamped_values():
    blockage_grid = BlockageGrid(radius=100, default_blockage_level=0.5)
    blockage_grid[Coord(1, 2)] = -0.1
    assert blockage_grid[Coord(1, 2)] == 0.0

    blockage_grid[Coord(3, 4)] = 1.5
    assert blockage_grid[Coord(3, 4)] == 1.0


@pytest.mark.parametrize('get_route', find_path_algorythms)
def test_get_route_without_penalties_fallbacks_to_default_route_calcilation(get_route):
    start = Coord(0, 0)
    end = Coord(2, 2)
    route = get_route(start, end)
    expected_route = start >> end
    assert route == expected_route


penalties_grids_test_cases = [
    # coordinates: start, end
    # penalties grid: {value: [(x, y), ...]}
    # expected route: []
    (
        ((0, 0), (4, 0)),
        {1.0: [(0, 1), (1, 0), (3, -1), (3, 0)]},
        ['↘', '→', '↗', '↗', '→', '↘'],
    ),
    (
        ((0, 0), (1, 1)),
        {1.0: [(-1, 1), (0, 1), (1, 0), (1, -1)]},
        ['←', '↖', '↗', '→', '→', '↘'],
    ),
    (
        ((0, 0), (1, 1)),
        {
            1.0: [(-1, 1), (0, 1), (1, 0), (1, -1), (-1, 0)],
            0.8: [(0, -1)],
        },
        ['↙', '↘', '→', '↗', '↗', '↖'],
    ),
    (
        ((0, 0), (1, 1)),
        {1.0: [(-1, 1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0)]},
        [],
    ),
]


@pytest.mark.parametrize('get_route', find_path_algorythms)
@pytest.mark.parametrize('start_end, penalties, expected', penalties_grids_test_cases)
def test_get_route_with_blockage_grid(get_route, start_end: tuple[Coord], penalties: BlockageGrid, expected: Route):
    start, end = start_end
    penalties_map = BlockageGrid(radius=100)
    for penalty_value, coordinates in penalties.items():
        for coord in coordinates:
            penalties_map[coord] = penalty_value

    route = get_route(start, end, penalties=penalties_map)
    expected_route = Route([Direction(d) for d in expected])
    assert route == expected_route


@pytest.mark.parametrize('get_route', find_path_algorythms)
def test_get_route_with_negative_step_penalty(get_route):
    with pytest.raises(ValueError):
        get_route((0, 0), (100, 100), penalties=BlockageGrid(1000), step_penalty=-0.5)


def test_get_blocked_areas_in_blockage_grid():
    """Test that all hexes out of grid radius are set to the maximum blockage value"""
    grid_radius = 30
    default_blockage_value = 0.2
    grid = BlockageGrid(radius=grid_radius, default_blockage_level=default_blockage_value)
    assert len(grid) == 0
    center = Coord(0, 0)
    for x, y in itertools.product(range(100), range(100)):
        if Coord(x, y) - center >= grid_radius:
            assert grid[x, y] == grid.MAX_VALUE
        else:
            assert grid[x, y] == default_blockage_value
