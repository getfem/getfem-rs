#!/usr/bin/python3
"""Writes the reference Exodus files of tests/exodus.rs with independent
tools (Debian: python3-meshio, python3-netcdf4, netcdf-bin), in the netCDF
64-bit offset format:
- meshio_2d.exo, meshio_3d.exo (meshio): blocks of linear elements,
  packed coordinates, a node set, a nodal field;
- sets.exo (netCDF4 directly, the Exodus II layout): 2 quads in the plane
  z = 0 of a 3D file, blocks 1 and 7, side sets 1 and 2, element set 1 (the
  id of side set 1), node set 5;
- quadratic.exo (netCDF4 directly): one element of each quadratic type,
  each an affine image of its reference element, nodes in the Exodus
  (Ioss) order.
Run from this directory: /usr/bin/python3 -I make_refs.py"""
import os
import subprocess

import meshio
import netCDF4
import numpy as np


def classic(src, dst):
  # meshio's file to the 64-bit offset format: no 64-bit integer
  # attribute, a fixed title (no date)
  with netCDF4.Dataset(src, "a") as nc:
    nc.delncattr("floating_point_word_size")
    nc.floating_point_word_size = np.int32(8)
    nc.title = "meshio"
  subprocess.run(["nccopy", "-k", "64-bit offset", src, dst], check=True)
  os.remove(src)


def with_meshio():
  i4 = np.int32
  # 2D: two triangles and a quad, a node set, a field at two steps
  p = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [2., 0.], [2., 1.]])
  m = meshio.Mesh(p, [("triangle", np.array([[0, 1, 2], [0, 2, 3]], i4)),
                      ("quad", np.array([[1, 4, 5, 2]], i4))],
                  point_sets={"left": np.array([0, 3], i4)},
                  point_data={"T": p[:, 0] + 2 * p[:, 1]})
  meshio.write("tmp.exo", m, file_format="exodus")
  classic("tmp.exo", "meshio_2d.exo")
  # 3D: a hexahedron, a wedge on top of it, a tetrahedron, a pyramid
  p = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
                [0.5, 0., 2.], [0.5, 1., 2.], [2., 0., 0.], [3., 0., 0.],
                [2., 1., 0.], [2., 0., 1.], [3., 1., 0.], [2.5, 0.5, 1.]])
  m = meshio.Mesh(p, [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], i4)),
                      ("wedge", np.array([[4, 5, 8, 7, 6, 9]], i4)),
                      ("tetra", np.array([[10, 11, 12, 13]], i4)),
                      ("pyramid", np.array([[10, 11, 14, 12, 15]], i4))],
                  point_sets={"top": np.array([8, 9], i4)})
  meshio.write("tmp.exo", m, file_format="exodus")
  classic("tmp.exo", "meshio_3d.exo")


def text(nc, name, dims, strings):
  v = nc.createVariable(name, "S1", dims)
  v.set_auto_mask(False)
  for i, s in enumerate(strings):
    for j, c in enumerate(s):
      v[i, j] = c.encode()


def header(nc, coords, nb_elem, nb_blk):
  nc.title = "GetFEM-rs reference"
  nc.version = np.float32(8.1)
  nc.api_version = np.float32(8.1)
  nc.floating_point_word_size = np.int32(8)
  nc.file_size = np.int32(1)
  nc.createDimension("len_string", 33)
  nc.createDimension("len_name", 33)
  nc.createDimension("len_line", 81)
  nc.createDimension("four", 4)
  nc.createDimension("time_step", None)
  nc.createDimension("num_dim", coords.shape[1])
  nc.createDimension("num_nodes", coords.shape[0])
  nc.createDimension("num_elem", nb_elem)
  nc.createDimension("num_el_blk", nb_blk)
  nc.createVariable("time_whole", "f8", ("time_step",))
  text(nc, "coor_names", ("num_dim", "len_name"), "XYZ"[:coords.shape[1]])
  for k, c in enumerate("xyz"[:coords.shape[1]]):
    nc.createVariable("coord" + c, "f8", ("num_nodes",))[:] = coords[:, k]


def block(nc, b, elem_type, conn):
  nc.createDimension("num_el_in_blk%d" % b, conn.shape[0])
  nc.createDimension("num_nod_per_el%d" % b, conn.shape[1])
  v = nc.createVariable("connect%d" % b, "i4",
                        ("num_el_in_blk%d" % b, "num_nod_per_el%d" % b))
  v.elem_type = elem_type
  v[:] = conn + 1


def with_sets():
  # quads 1 = (0,0)-(1,1) in block 1, 2 = (1,0)-(2,1) in block 7
  x = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]])
  nc = netCDF4.Dataset("sets.exo", "w", format="NETCDF3_64BIT_OFFSET")
  header(nc, x, 2, 2)
  nc.createVariable("eb_prop1", "i4", ("num_el_blk",))[:] = [1, 7]
  block(nc, 1, "QUAD4", np.array([[0, 1, 2, 3]]))
  block(nc, 2, "SHELL4", np.array([[1, 4, 5, 2]]))
  # side set 1: the left edge of element 1 (edge 4), side set 2: the right
  # edge of element 2 (edge 2); quads in a 3D file are shells: sides 1, 2
  # their faces, the edges from side 3
  nc.createDimension("num_side_sets", 2)
  nc.createVariable("ss_prop1", "i4", ("num_side_sets",))[:] = [1, 2]
  for k, (e, s) in enumerate([(1, 6), (2, 4)]):
    nc.createDimension("num_side_ss%d" % (k + 1), 1)
    nc.createVariable("elem_ss%d" % (k + 1), "i4", ("num_side_ss%d" % (k + 1),))[:] = [e]
    nc.createVariable("side_ss%d" % (k + 1), "i4", ("num_side_ss%d" % (k + 1),))[:] = [s]
  # element set 1: element 2
  nc.createDimension("num_elem_sets", 1)
  nc.createVariable("els_prop1", "i4", ("num_elem_sets",))[:] = [1]
  nc.createDimension("num_ele_els1", 1)
  nc.createVariable("elem_els1", "i4", ("num_ele_els1",))[:] = [2]
  # node set 5: the left nodes
  nc.createDimension("num_node_sets", 1)
  nc.createVariable("ns_prop1", "i4", ("num_node_sets",))[:] = [5]
  nc.createDimension("num_nod_ns1", 2)
  nc.createVariable("node_ns1", "i4", ("num_nod_ns1",))[:] = [1, 4]
  nc.close()


# vertices of the reference elements (Exodus order) and the extra nodes,
# each the centre of the vertices listed (0-based, from Ioss' edge and face
# tables)
LINE = [[0], [1]]
TRI = [[0, 0], [1, 0], [0, 1]]
QUAD = [[0, 0], [1, 0], [1, 1], [0, 1]]
TET = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
HEX = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
WEDGE = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]
PYR = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]]
QUAD_E = [[0, 1], [1, 2], [2, 3], [3, 0]]
HEX_E = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]
WEDGE_E = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 4], [2, 5], [3, 4], [4, 5], [5, 3]]
PYR_E = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
TYPES = [
  ("BAR3", LINE, [[0, 1]]),
  ("TRI6", TRI, [[0, 1], [1, 2], [2, 0]]),
  ("QUAD8", QUAD, QUAD_E),
  ("QUAD9", QUAD, QUAD_E + [[0, 1, 2, 3]]),
  ("TETRA10", TET, [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]),
  ("HEX20", HEX, HEX_E),
  ("HEX27", HEX, HEX_E + [list(range(8)), [0, 3, 2, 1], [4, 5, 6, 7], [0, 4, 7, 3],
                          [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6]]),
  ("WEDGE15", WEDGE, WEDGE_E),
  ("WEDGE18", WEDGE, WEDGE_E + [[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]]),
  ("PYRAMID13", PYR, PYR_E),
  ("PYRAMID14", PYR, PYR_E + [[0, 1, 2, 3]]),
]


def with_quadratic():
  # each element: x = A r + b in 3D (lines and faces embedded), one block each
  rng = np.random.default_rng(1)
  coords, conns = [], []
  for k, (name, verts, extra) in enumerate(TYPES):
    v = np.array(verts, float)
    r = np.vstack([v] + [v[e].mean(axis=0) for e in extra])
    A = np.eye(3)[:, :v.shape[1]] + 0.2 * rng.random((3, v.shape[1]))
    x = r @ A.T + [3. * k, 0, 0]
    conns.append(sum(map(len, coords)) + np.arange(len(r)))
    coords.append(x)
  x = np.vstack(coords)
  nc = netCDF4.Dataset("quadratic.exo", "w", format="NETCDF3_64BIT_OFFSET")
  header(nc, x, len(TYPES), len(TYPES))
  nc.createVariable("eb_prop1", "i4", ("num_el_blk",))[:] = range(1, len(TYPES) + 1)
  for k, (name, _, _) in enumerate(TYPES):
    block(nc, k + 1, name, conns[k][None, :])
  nc.close()


if __name__ == "__main__":
  with_meshio()
  with_sets()
  with_quadratic()
