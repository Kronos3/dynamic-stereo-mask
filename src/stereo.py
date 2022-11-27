# Python imports
from abc import ABC, abstractmethod
from pathlib import Path

# Library imports
import cv2
import numpy as np
import pandas as pd
import open3d as o3d

# Project imports
from src.kitti import KittiFrame


class Stereo(ABC):
    def __init__(self, calibration: Path):
        calib = pd.read_csv(calibration, delimiter=' ',
                            header=None, index_col=0)

        self.P2 = np.array(calib.loc["P2:"]).reshape((3, 4))
        self.P3 = np.array(calib.loc["P3:"]).reshape((3, 4))

        self.k_left, self.r_left, self.t_left = Stereo._decompose_projection_matrix(
            self.P2)
        self.k_right, self.r_right, self.t_right = Stereo._decompose_projection_matrix(
            self.P3)

        self.left_instrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.left_instrinsic.intrinsic_matrix = self.k_left
        self.left_extrinsic = np.vstack(
            (np.hstack((self.r_left, self.t_left[:])), [0, 0, 0, 1]))

    @staticmethod
    def _decompose_projection_matrix(p):
        k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
        t = (t / t[3])[:3]

        return k, r, t

    @abstractmethod
    def disparity(self, frame: KittiFrame):
        """
        Compute the disparity between the left and right images
        :param frame: kitti image frame
        :return: disparity map
        """
        ...

    def depth(self, disp, rectified=True) -> cv2.Mat:
        # Get focal length of x axis for left camera
        f = self.k_left[0][0]

        # Calculate baseline of stereo pair
        if rectified:
            b = self.t_right[0] - self.t_left[0]
        else:
            b = self.t_left[0] - self.t_right[0]

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1

        # Make empty depth map then fill with depth
        depth_map = np.ones(disp.shape)
        depth_map = f * b / disp

        # Mask out the left side of the image
        # This is caused by the right camera not seeing this portion of the image
        mask = np.zeros(disp.shape[:2], dtype=np.uint8)
        ymax = disp.shape[0]
        xmax = disp.shape[1]
        cv2.rectangle(mask, (96, 0), (xmax, ymax), 255, thickness=-1)

        depth_map[mask] = 0
        return depth_map

    def point_cloud(self, frame: KittiFrame, depth: cv2.Mat):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(frame.left_color().astype(np.uint8)),
            o3d.geometry.Image(depth.astype(np.float32))
        )

        return o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.left_instrinsic,
            self.left_extrinsic
        )


class OCVStereo(Stereo, ABC):
    matcher: any

    def __init__(self, calibration: Path):
        super().__init__(calibration)

    def disparity(self, frame: KittiFrame):
        return self.matcher.compute(frame.left_gray(), frame.right_gray()).astype(np.float32) / 16


class SGBMStereo(OCVStereo):
    def __init__(self, calibration: Path):
        super().__init__(calibration)

        sad_window = 6
        num_disparities = sad_window * 16
        self.matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                             blockSize=11,
                                             P1=8 * 3 * sad_window ** 2,
                                             P2=32 * 3 * sad_window ** 2,
                                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


class BMStereo(OCVStereo):

    def __init__(self, calibration: Path):
        super().__init__(calibration)

        sad_window = 6
        num_disparities = sad_window * 16
        self.matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                           blockSize=11)
