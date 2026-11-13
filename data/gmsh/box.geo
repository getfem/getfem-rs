SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 0.5, 0.5};
Mesh.MeshSizeMin = 0.3; Mesh.MeshSizeMax = 0.6;
Physical Volume("solid", 1) = {1};
Physical Surface("xmin", 2) = {1};
Physical Surface("xmax", 3) = {2};
