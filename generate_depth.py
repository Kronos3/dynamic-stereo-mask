from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src import kitti
from src.lea_stereo import LEAStereo

dataset = kitti.KittiDataset(Path('dataset/sequences/00'))

ster = LEAStereo(dataset.calib)

for index in tqdm(range(int(sys.argv[1]), int(sys.argv[2]))):
    frame = kitti.KittiFrame(dataset, index)
    disp = ster.disparity(frame)
    depth = ster.depth(disp)

    plt.imsave("out/disp%d.png" % index, disp)
    np.save("depth/depth%d" % index, depth)
