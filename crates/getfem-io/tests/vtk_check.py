#!/usr/bin/env python3
"""Check exported VTK files with VTK itself: for each cell, VTK's own
interpolation at given parametric points must reproduce the positions
computed by GetFEM-rs. Usage: vtk_check.py (<file.vtu|.vtk> <points.txt>)...
points.txt: one line per cell: vtk_type r s t x y z (repeated)."""
import sys
import vtk


def check(f, ref):
  rd = vtk.vtkXMLUnstructuredGridReader() if f.endswith(".vtu") else vtk.vtkUnstructuredGridReader()
  rd.SetFileName(f)
  rd.Update()
  g = rd.GetOutput()
  lines = [l.split() for l in open(ref) if l.strip()]
  assert g.GetNumberOfCells() == len(lines), (g.GetNumberOfCells(), len(lines))
  worst = 0.0
  for i, l in enumerate(lines):
    c = g.GetCell(i)
    assert c.GetCellType() == int(l[0]), (i, c.GetCellType(), l[0])
    vals = list(map(float, l[1:]))
    for k in range(0, len(vals), 6):
      pc, x = vals[k:k + 3], vals[k + 3:k + 6]
      y = [0.0] * 3
      w = [0.0] * c.GetNumberOfPoints()
      c.EvaluateLocation(vtk.reference(0), pc, y, w)
      worst = max(worst, max(abs(a - b) for a, b in zip(x, y)))
  print(f"{f}: {len(lines)} cells, max deviation {worst:.2e}")
  return worst < 1e-9


args = sys.argv[1:]
ok = all([check(args[i], args[i + 1]) for i in range(0, len(args), 2)])
sys.exit(0 if ok else 1)
