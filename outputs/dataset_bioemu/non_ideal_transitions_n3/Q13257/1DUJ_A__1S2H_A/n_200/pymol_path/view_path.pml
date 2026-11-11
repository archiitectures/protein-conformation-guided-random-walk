reinitialize
bg_color white
load 00_A.pdb, path_00_A
hide everything, path_00_A
show spheres, path_00_A
set sphere_scale, 0.45, path_00_A
color marine, path_00_A
load 01_bioemu_000046.pdb, path_01_bioemu_000046
hide everything, path_01_bioemu_000046
show spheres, path_01_bioemu_000046
set sphere_scale, 0.45, path_01_bioemu_000046
color cyan, path_01_bioemu_000046
load 02_bioemu_000075.pdb, path_02_bioemu_000075
hide everything, path_02_bioemu_000075
show spheres, path_02_bioemu_000075
set sphere_scale, 0.45, path_02_bioemu_000075
color tv_green, path_02_bioemu_000075
load 03_bioemu_000027.pdb, path_03_bioemu_000027
hide everything, path_03_bioemu_000027
show spheres, path_03_bioemu_000027
set sphere_scale, 0.45, path_03_bioemu_000027
color yellow, path_03_bioemu_000027
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
