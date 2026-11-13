// prisms by extrusion of a triangle mesh
Point(1) = {0, 0, 0, 0.5}; Point(2) = {1, 0, 0, 0.5}; Point(3) = {0, 1, 0, 0.5};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 1};
Curve Loop(1) = {1, 2, 3}; Plane Surface(1) = {1};
out[] = Extrude {0, 0, 0.6} { Surface{1}; Layers{2}; Recombine; };
Physical Volume("solid", 1) = {out[1]};
Physical Surface("bottom", 2) = {1};
Physical Surface("top", 3) = {out[0]};
