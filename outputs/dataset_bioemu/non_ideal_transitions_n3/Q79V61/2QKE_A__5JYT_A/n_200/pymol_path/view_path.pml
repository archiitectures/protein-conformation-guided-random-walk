reinitialize
bg_color white
load 00_A.pdb, path_00_A
hide everything, path_00_A
show spheres, path_00_A
set sphere_scale, 0.45, path_00_A
color marine, path_00_A
load 01_bioemu_000055.pdb, path_01_bioemu_000055
hide everything, path_01_bioemu_000055
show spheres, path_01_bioemu_000055
set sphere_scale, 0.45, path_01_bioemu_000055
color cyan, path_01_bioemu_000055
load 02_bioemu_000120.pdb, path_02_bioemu_000120
hide everything, path_02_bioemu_000120
show spheres, path_02_bioemu_000120
set sphere_scale, 0.45, path_02_bioemu_000120
color tv_green, path_02_bioemu_000120
load 03_B.pdb, path_03_B
hide everything, path_03_B
show spheres, path_03_B
set sphere_scale, 0.45, path_03_B
color yellow, path_03_B
set ray_opaque_background, off
set depth_cue, 0
orient
zoom
png path_nodes.png, width=1600, height=1000, ray=1
