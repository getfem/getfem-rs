// unit square, physical boundary parts and two surfaces
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1};
Rectangle(2) = {1, 0, 0, 0.5, 1};
BooleanFragments{ Surface{1}; Delete; }{ Surface{2}; Delete; }
Mesh.CharacteristicLengthMax = 0.35;
Physical Surface("left_part", 11) = {1};
Physical Surface("right_part", 12) = {2};
Physical Curve("bottom", 21) = {1, 5};
Physical Curve("top", 22) = {3, 7};
Physical Curve("inner", 23) = {2};
