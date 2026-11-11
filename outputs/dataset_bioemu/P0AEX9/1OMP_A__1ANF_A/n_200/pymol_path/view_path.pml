reinitialize
bg_color white
load 00_A.pdb, path_00_A
hide everything, path_00_A
show spheres, path_00_A
set sphere_scale, 0.45, path_00_A
color marine, path_00_A
load 01_bioemu_000049.pdb, path_01_bioemu_000049
hide everything, path_01_bioemu_000049
show spheres, path_01_bioemu_000049
set sphere_scale, 0.45, path_01_bioemu_000049
color cyan, path_01_bioemu_000049
load 02_bioemu_000150.pdb, path_02_bioemu_000150
hide everything, path_02_bioemu_000150
show spheres, path_02_bioemu_000150
set sphere_scale, 0.45, path_02_bioemu_000150
color tv_green, path_02_bioemu_000150
load 03_bioemu_000098.pdb, path_03_bioemu_000098
hide everything, path_03_bioemu_000098
show spheres, path_03_bioemu_000098
set sphere_scale, 0.45, path_03_bioemu_000098
color yellow, path_03_bioemu_000098
load 04_bioemu_000158.pdb, path_04_bioemu_000158
hide everything, path_04_bioemu_000158
show spheres, path_04_bioemu_000158
set sphere_scale, 0.45, path_04_bioemu_000158
color orange, path_04_bioemu_000158
load 05_bioemu_000085.pdb, path_05_bioemu_000085
hide everything, path_05_bioemu_000085
show spheres, path_05_bioemu_000085
set sphere_scale, 0.45, path_05_bioemu_000085
color salmon, path_05_bioemu_000085
load 06_bioemu_000115.pdb, path_06_bioemu_000115
hide everything, path_06_bioemu_000115
show spheres, path_06_bioemu_000115
set sphere_scale, 0.45, path_06_bioemu_000115
color red, path_06_bioemu_000115
load 07_bioemu_000091.pdb, path_07_bioemu_000091
hide everything, path_07_bioemu_000091
show spheres, path_07_bioemu_000091
set sphere_scale, 0.45, path_07_bioemu_000091
color magenta, path_07_bioemu_000091
load 08_B.pdb, path_08_B
hide everything, path_08_B
show spheres, path_08_B
set sphere_scale, 0.45, path_08_B
color purpleblue, path_08_B
set ray_opaque_background, off
set depth_cue, 0
orient
zoom
png path_nodes.png, width=1600, height=1000, ray=1
