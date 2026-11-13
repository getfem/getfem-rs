#!/usr/bin/python3
"""Reads an Exodus file with meshio (an independent reader) and prints
its number of points, its cells (type count) and its nodal fields."""
import sys

import meshio

m = meshio.read(sys.argv[1], file_format="exodus")
print(len(m.points))
print(" ".join("%s:%d" % (c.type, len(c.data)) for c in m.cells))
print(" ".join(sorted(m.point_data)))
