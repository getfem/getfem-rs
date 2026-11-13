// structured hexahedra
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 0.5, 0.5};
Transfinite Curve{:} = 3;
Transfinite Surface{:};
Recombine Surface{:};
Transfinite Volume{1};
Physical Volume("solid", 1) = {1};
Physical Surface("xmin", 2) = {1};
Physical Surface("xmax", 3) = {2};
