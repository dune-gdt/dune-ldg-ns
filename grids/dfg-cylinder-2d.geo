// DFG benchmark geometry (Schaefer/Turek 1996): channel [0, 2.2] x [0, 0.41] minus the disc of radius 0.05 at (0.2, 0.2).
// Generate with:  gmsh -2 -format msh22 -setnumber h 0.02 -setnumber hc 0.005 dfg-cylinder-2d.geo -o dfg-cylinder-2d.msh
// (Dune::GmshReader in DUNE 2.10 reads the MSH 2.2 format.)
// NOTE: the grid is affine, the cylinder is approximated by straight edges of length ~hc. For high-order runs keep hc
// small (the geometric error of the polygonal cylinder dominates c_D otherwise), see doc/design.md, Sec. 5.3.
If (!Exists(h))
  h = 0.02;   // far-field mesh size
EndIf
If (!Exists(hc))
  hc = 0.005; // mesh size on the cylinder
EndIf
L = 2.2; H = 0.41; cx = 0.2; cy = 0.2; r = 0.05;

Point(1) = {0, 0, 0, h};
Point(2) = {L, 0, 0, h};
Point(3) = {L, H, 0, h};
Point(4) = {0, H, 0, h};
Point(5) = {cx, cy, 0, hc};
Point(6) = {cx + r, cy, 0, hc};
Point(7) = {cx, cy + r, 0, hc};
Point(8) = {cx - r, cy, 0, hc};
Point(9) = {cx, cy - r, 0, hc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};

// refine the wake region
Field[1] = Box;
Field[1].VIn = 2 * hc;
Field[1].VOut = h;
Field[1].XMin = 0.1;
Field[1].XMax = 1.0;
Field[1].YMin = 0.1;
Field[1].YMax = 0.3;
Background Field = 1;

Physical Curve("inflow", 1) = {4};
Physical Curve("outflow", 2) = {2};
Physical Curve("walls", 3) = {1, 3};
Physical Curve("cylinder", 4) = {5, 6, 7, 8};
Physical Surface("fluid", 1) = {1};
Mesh.MshFileVersion = 2.2;
