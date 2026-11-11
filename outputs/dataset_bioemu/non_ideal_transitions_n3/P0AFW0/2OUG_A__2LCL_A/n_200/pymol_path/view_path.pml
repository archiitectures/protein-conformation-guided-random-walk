reinitialize
bg_color white
load 00_A.pdb, path_00_A
hide everything, path_00_A
show spheres, path_00_A
set sphere_scale, 0.45, path_00_A
color marine, path_00_A
load 01_bioemu_000071.pdb, path_01_bioemu_000071
hide everything, path_01_bioemu_000071
show spheres, path_01_bioemu_000071
set sphere_scale, 0.45, path_01_bioemu_000071
color cyan, path_01_bioemu_000071
load 02_bioemu_000162.pdb, path_02_bioemu_000162
hide everything, path_02_bioemu_000162
show spheres, path_02_bioemu_000162
set sphere_scale, 0.45, path_02_bioemu_000162
color tv_green, path_02_bioemu_000162
load 03_bioemu_000023.pdb, path_03_bioemu_000023
hide everything, path_03_bioemu_000023
show spheres, path_03_bioemu_000023
set sphere_scale, 0.45, path_03_bioemu_000023
color yellow, path_03_bioemu_000023
load 04_B.pdb, path_04_B
hide everything, path_04_B
show spheres, path_04_B
set sphere_scale, 0.45, path_04_B
color orange, path_04_B
set ray_opaque_background, off
set depth_cue, 0
orient
zoom
png path_nodes.png, width=1600, height=1000, ray=1
