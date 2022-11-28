from pathlib import Path

import open3d as o3d

from src import kitti, stereo
from src.lea_stereo import LEAStereo

dataset = kitti.KittiDataset(Path('dataset/sequences/00'))

test_frame = kitti.KittiFrame(dataset, 654)
ster = LEAStereo(dataset.calib)

disp = ster.disparity(test_frame)
depth = ster.depth(disp)

pt_cloud = ster.point_cloud(test_frame, depth)
o3d.visualization.draw_geometries([pt_cloud])
