#!/usr/bin/env bash
# regenerate the gmsh test meshes (msh 2.2 and 4.1)
set -euo pipefail
cd "$(dirname "$0")"
g() { gmsh "$1" -"$2" -order "$3" $4 -format "$5" -o "$6" -v 0; }
for f in msh22 msh41; do
  g square.geo 2 1 "" $f square_tri1_$f.msh
  g square.geo 2 2 "" $f square_tri2_$f.msh
  g square.geo 2 3 "" $f square_tri3_$f.msh
  g square.geo 2 1 "-setnumber Mesh.RecombineAll 1" $f square_quad1_$f.msh
  g square.geo 2 2 "-setnumber Mesh.RecombineAll 1" $f square_quad2_$f.msh
  g square.geo 2 2 "-setnumber Mesh.RecombineAll 1 -setnumber Mesh.SecondOrderIncomplete 1" $f square_quad8_$f.msh
  g box.geo 3 1 "" $f box_tet1_$f.msh
  g box.geo 3 2 "" $f box_tet2_$f.msh
  g hexbox.geo 3 1 "" $f box_hex1_$f.msh
  g hexbox.geo 3 2 "" $f box_hex2_$f.msh
  g hexbox.geo 3 2 "-setnumber Mesh.SecondOrderIncomplete 1" $f box_hex20_$f.msh
  g prism.geo 3 1 "" $f prism1_$f.msh
  g prism.geo 3 2 "" $f prism2_$f.msh
done
