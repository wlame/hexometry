import pytest

from hexometry.coordinates import Coord, Direction, Route
from hexometry.grids import Grid, BlockageGrid, dijkstra


def test_grid_repr():
    grid = Grid(default=None)
    assert repr(grid) == '<Grid({})>'


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
    blockage_grid = BlockageGrid(default_blockage_level=0.5)
    assert blockage_grid.default == 0.5
    assert blockage_grid.get_default is None


def test_blockagegrid_normalize_clamps_values():
    blockage_grid = BlockageGrid()
    assert blockage_grid.normalize(Coord(1, 2), -0.1) == 0.0
    assert blockage_grid.normalize(Coord(1, 2), 1.5) == 1.0
    assert blockage_grid.normalize(Coord(1, 2), 0.7) == 0.7


def test_blockagegrid_getitem_returns_clamped_values():
    blockage_grid = BlockageGrid(default_blockage_level=0.5)
    blockage_grid[Coord(1, 2)] = -0.1
    assert blockage_grid[Coord(1, 2)] == 0.0

    blockage_grid[Coord(3, 4)] = 1.5
    assert blockage_grid[Coord(3, 4)] == 1.0


def test_dijkstra_without_penalties_fallbacks_to_default_route_calcilation():
    start = Coord(0, 0)
    end = Coord(2, 2)
    route = dijkstra(start, end)
    expected_route = start >> end
    assert route == expected_route


penalties_grids_test_cases = [
    # coordinates: start, end
    # penalties grid: {value: [(x, y), ...]}
    # expected route: []
    (
        (0, 0), (4, 0),
        {1.0: [(0,1), (1, 0), (3, -1), (3, 0)]},
        ['↘', '→', '↗', '↗', '→', '↘'],
    ),
    (
        (0, 0), (1, 1),
        {1.0: [(-1,1), (0, 1), (1, 0), (1, -1)]},
        ['←', '↖', '↗', '→', '→', '↘'],
    ),
    (
        (0, 0), (1, 1),
        {
            1.0: [(-1,1), (0, 1), (1, 0), (1, -1), (-1, 0)],
            0.8: [(0, -1)],
        },
        ['↙', '↘', '→', '↗', '↗', '↖'],
    ),
    (
        (0, 0), (1, 1),
        {1.0: [(-1,1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0)]},
        [],
    ),
]

@pytest.mark.parametrize('start, end, penalties, expected', penalties_grids_test_cases)
def test_dijkstra_with_blockage_grid(start: Coord, end: Coord, penalties: BlockageGrid, expected: Route):
    penalties_map = BlockageGrid()
    for penalty_value, coordinates in penalties.items():
        for coord in coordinates:
            penalties_map[coord] = penalty_value

    route = dijkstra(start, end, penalties=penalties_map)
    expected_route = Route([Direction(d) for d in expected])
    assert route == expected_route


def test_dijkstra_with_negative_step_penalty():
    with pytest.raises(ValueError):
        dijkstra((0,0), (100, 100), penalties=BlockageGrid(), step_penalty=-0.5)
