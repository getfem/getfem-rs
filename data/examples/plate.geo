// plate with three holes; outer boundary 1, holes 2
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 3, 2};
Disk(2) = {0.8, 1.0, 0, 0.35};
Disk(3) = {1.9, 1.35, 0, 0.25};
Disk(4) = {2.3, 0.6, 0, 0.18};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2, 3, 4}; Delete; }
Mesh.MeshSizeMax = 0.09;
Mesh.ElementOrder = 2;
Physical Surface("plate", 1) = {1};
e = 1e-3;
outer[] = Curve In BoundingBox{-e, -e, -1, 3 + e, e, 1};
outer[] += Curve In BoundingBox{-e, 2 - e, -1, 3 + e, 2 + e, 1};
outer[] += Curve In BoundingBox{-e, -e, -1, e, 2 + e, 1};
outer[] += Curve In BoundingBox{3 - e, -e, -1, 3 + e, 2 + e, 1};
Physical Curve("outer", 10) = {outer[]};
Physical Curve("holes", 20) = Curve In BoundingBox{0.4, 0.3, -1, 2.6, 1.7, 1};
