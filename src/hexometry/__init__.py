"""
**Hexometry**

A Python library for working with hexagonal grids, providing tools for coordinate manipulation, pathfinding, and grid operations.

Key Features:
- Hexagonal coordinate system conversions
- Neighbor finding and distance calculations
- Route generation between coordinates
- Pathfinding using Dijkstra's algorithm with optional penalties
- Grid management for hex-based applications
"""

from .coordinates import (
    Direction,
    Coord,
    Route,
    DecartCoord,
)
from .grids import (
    Grid,
    BlockageGrid,
    dijkstra,
)

__version__ = '1.1.1'


__all__ = [
    Direction,
    Coord,
    Route,
    DecartCoord,
    Grid,
    BlockageGrid,
    dijkstra,
]
