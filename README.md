# Hexometry
Python library to work with hexagon grid geometry.

## Overview
Hexometry is a Python library to work with hexagon grid geometry.
Some ideas of having x, y, z coordinates for hexagon grid are taken from [this article](https://catlikecoding.com/unity/tutorials/hex-map/part-1/) by Jasper Flick.

Particulary this image gives a good idea of a choosen coordinates for hexagon grid:

![hex grid coordinates example](https://catlikecoding.com/unity/tutorials/hex-map/part-1/hexagonal-coordinates/cube-diagram.png)

## Dependencies
This library has no dependencies apart from python stdlib, but it requires Python 3.10 or higher.
There is no problem to make it work with Python 3.7 or 3.8, but the original intention was to try fancy features of modern Python.

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
