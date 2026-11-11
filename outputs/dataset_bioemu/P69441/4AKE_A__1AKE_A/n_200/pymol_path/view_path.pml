reinitialize
bg_color white
load 00_A.pdb, path_00_A
hide everything, path_00_A
show spheres, path_00_A
set sphere_scale, 0.45, path_00_A
color marine, path_00_A
load 01_bioemu_000084.pdb, path_01_bioemu_000084
hide everything, path_01_bioemu_000084
show spheres, path_01_bioemu_000084
set sphere_scale, 0.45, path_01_bioemu_000084
color cyan, path_01_bioemu_000084
load 02_bioemu_000164.pdb, path_02_bioemu_000164
hide everything, path_02_bioemu_000164
show spheres, path_02_bioemu_000164
set sphere_scale, 0.45, path_02_bioemu_000164
color tv_green, path_02_bioemu_000164
load 03_bioemu_000088.pdb, path_03_bioemu_000088
hide everything, path_03_bioemu_000088
show spheres, path_03_bioemu_000088
set sphere_scale, 0.45, path_03_bioemu_000088
color yellow, path_03_bioemu_000088
load 04_bioemu_000174.pdb, path_04_bioemu_000174
hide everything, path_04_bioemu_000174
show spheres, path_04_bioemu_000174
set sphere_scale, 0.45, path_04_bioemu_000174
color orange, path_04_bioemu_000174
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
