reinitialize
bg_color white
load 00_A.pdb, path_00_A
hide everything, path_00_A
show spheres, path_00_A
set sphere_scale, 0.45, path_00_A
color marine, path_00_A
load 01_bioemu_000124.pdb, path_01_bioemu_000124
hide everything, path_01_bioemu_000124
show spheres, path_01_bioemu_000124
set sphere_scale, 0.45, path_01_bioemu_000124
color cyan, path_01_bioemu_000124
load 02_bioemu_000089.pdb, path_02_bioemu_000089
hide everything, path_02_bioemu_000089
show spheres, path_02_bioemu_000089
set sphere_scale, 0.45, path_02_bioemu_000089
color tv_green, path_02_bioemu_000089
load 03_bioemu_000079.pdb, path_03_bioemu_000079
hide everything, path_03_bioemu_000079
show spheres, path_03_bioemu_000079
set sphere_scale, 0.45, path_03_bioemu_000079
color yellow, path_03_bioemu_000079
load 04_bioemu_000112.pdb, path_04_bioemu_000112
hide everything, path_04_bioemu_000112
show spheres, path_04_bioemu_000112
set sphere_scale, 0.45, path_04_bioemu_000112
color orange, path_04_bioemu_000112
load 05_B.pdb, path_05_B
hide everything, path_05_B
show spheres, path_05_B
set sphere_scale, 0.45, path_05_B
color salmon, path_05_B
set ray_opaque_background, off
set depth_cue, 0
orient
zoom
png path_nodes.png, width=1600, height=1000, ray=1
