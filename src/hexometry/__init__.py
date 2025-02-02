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

from .coordinates import *
from .grids import *

__version__ = '1.0.1'
