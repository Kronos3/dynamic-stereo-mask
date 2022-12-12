from pathlib import Path
import sys

import open3d as o3d
import numpy as np

from tqdm import tqdm

from src import kitti, stereo

dataset = kitti.KittiDataset(Path('dataset/sequences/00'))

transform = np.identity(4)
last_cloud = None

WITH_MASK = True

ster = stereo.SGBMStereo(dataset.calib)

start, end = int(sys.argv[1]), int(sys.argv[2])

odometry = []
for index in tqdm(range(start, end)):
    frame = kitti.KittiFrame(dataset, index)

    depth = np.load("depth/depth%d.npy" % index).astype(np.float32)

    # Get rid of out the things
    if WITH_MASK:
        mask = np.load("masks/mask%d.npy" % index).astype(np.bool8)
        depth[mask] = 0

    depth[depth > 0.45 * depth.max()] = 0

    pt_cloud = ster.point_cloud(frame, depth)
    cl, ind = pt_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pt_cloud = pt_cloud.select_by_index(ind)

    if last_cloud is None:
        full_cloud = pt_cloud
    else:
        # Point to Plane requires normal vectors for plane estimation
        last_cloud.estimate_normals()

        # Compute the transform between the current car frame and the world frame (first frame)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pt_cloud, last_cloud, 0.0002, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # Place the point cloud in the original coordinate frame
        pt_cloud.transform(reg_p2p.transformation)

        # Update the ICP estimation
        transform = reg_p2p.transformation

    last_cloud = pt_cloud
    odometry.append(transform)

pt_file = f"full-{start}-{end}{'mask' if WITH_MASK else 'no-mask'}.odom"
print("Saving to %s" % pt_file)
with open(pt_file, "wb+") as f:
    np.save(f, odometry)
