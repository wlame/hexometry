# Hexometry
Python library to work with hexagon grid geometry.

## Overview
Hexometry is a Python library to work with hexagon grid geometry.
Some ideas of having x, y, z coordinates for hexagon grid are taken from [this article](https://catlikecoding.com/unity/tutorials/hex-map/part-1/) by Jasper Flick.

Particulary this image gives a good idea of a choosen coordinates for hexagon grid:

![hex grid coordinates example](https://catlikecoding.com/unity/tutorials/hex-map/part-1/hexagonal-coordinates/cube-diagram.png)

### Features
  1. Hex cell coordinates:
    - find neighbour hexes
    - calculate distance between hexes
    - calculate coordinates in decart grid (do draw on regular x:y plot) respecting scale factor
    - calculate the shortest `Route` between two hex coordinates
    - move hex in any direction or along the `Route`
  2. Direction — Enum of 6 possible values `↗→↘↙←↖` matched to the sides of the world from NE to NW
  3. Route — list of `Directions`, that is a result of any path search
  4. Grids:
    - `Grid` — base grid class that represents a grid of hex coordinates with some values aggigned to each
    - `BlockageGrid` — hex grid of float blockage values from 0.0 to 1.0 that can be used in finding optimal paths
    - `dijkstra` and `a_star` algorythms of path finding on a blockage grids respecting penalties on hexes

## Dependencies
This library has no dependencies apart from python stdlib, but it requires Python 3.11 or higher.
There is no problem to make it work with Python 3.7 — 3.11, but the original intention was to try fancy features of modern Python.

## Installation
```bash
pip install hexometry
```

## Usage
```python
from hexometry import Coord, Direction, Route

c1 = Coord(0, 0)
c2 = Coord(2, 3)
print(f"Coordinates {c1} and {c2} on a field are {c1-c2} hexes away each other")

route = c1 >> c2
print(f"and the route between them are: {route}. Let's traverse it:")

coord = c1
for direction in route:
    coord += direction
    print(f"moving {direction.value} to {coord}")

assert coord == c2
print("here we are")
```
