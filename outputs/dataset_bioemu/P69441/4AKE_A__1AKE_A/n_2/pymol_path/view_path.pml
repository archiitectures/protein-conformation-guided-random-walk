reinitialize
bg_color white
load 00_A.pdb, path_00_A
hide everything, path_00_A
show spheres, path_00_A
set sphere_scale, 0.45, path_00_A
color marine, path_00_A
load 01_bioemu_000000.pdb, path_01_bioemu_000000
hide everything, path_01_bioemu_000000
show spheres, path_01_bioemu_000000
set sphere_scale, 0.45, path_01_bioemu_000000
color cyan, path_01_bioemu_000000
load 02_B.pdb, path_02_B
hide everything, path_02_B
show spheres, path_02_B
set sphere_scale, 0.45, path_02_B
color tv_green, path_02_B
set ray_opaque_background, off
set depth_cue, 0
orient
zoom
png path_nodes.png, width=1600, height=1000, ray=1
