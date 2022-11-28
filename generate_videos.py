from pathlib import Path

import open3d as o3d
import cv2

from src import kitti
from src.lea_stereo import LEAStereo

dataset = kitti.KittiDataset(Path('dataset/sequences/00'))

frame_range = range(600, 700)

size_test_frame = kitti.KittiFrame(dataset, 0)

raw_video = cv2.VideoWriter('raw.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, size_test_frame.left_color())
disparity_video = cv2.VideoWriter('disparity.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, size_test_frame.left_color())

for index in frame_range:
    frame = kitti.KittiFrame(dataset, index)
    ster = LEAStereo(dataset.calib)

    raw_video.write(frame.left_color())

    disp = ster.disparity(frame)

    colormapped_disparity = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    disparity_video.write(colormapped_disparity)

    # depth = ster.depth(disp)

    # pt_cloud = ster.point_cloud(frame, depth)

