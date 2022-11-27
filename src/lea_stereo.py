from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable

from src.kitti import KittiFrame
from src.stereo import Stereo
from leastereo.retrain.LEAStereo import LEAStereo as LEAStereoModel

CURRENT_DIR = Path(__file__).parent


class LEAStereo(Stereo):
    class Args:
        def __init__(self, args):
            self.args = args

        def __getattr__(self, item):
            return self.args[item]

    def __init__(self, calibration: Path):
        super().__init__(calibration)

        trained_models = CURRENT_DIR.parent / 'leastereo'
        self.args = LEAStereo.Args({
            "cuda": True,
            "crop_height": 384,
            "crop_width": 1248,
            "maxdisp": 192,
            "fea_num_layers": 6,
            "mat_num_layers": 12,
            "fea_filter_multiplier": 8,
            "fea_block_multiplier": 4,
            "fea_step": 3,
            "mat_filter_multiplier": 8,
            "mat_block_multiplier": 4,
            "mat_step": 3,
            "net_arch_fea": str(trained_models / 'run/sceneflow/best/architecture/feature_network_path.npy'),
            "cell_arch_fea": str(trained_models / 'run/sceneflow/best/architecture/feature_genotype.npy'),
            "net_arch_mat": str(trained_models / 'run/sceneflow/best/architecture/matching_network_path.npy'),
            "cell_arch_mat": str(trained_models / 'run/sceneflow/best/architecture/matching_genotype.npy'),
            "resume":  str(trained_models / 'run/Kitti15/best/best.pth')
        })

        self.model = LEAStereoModel(self.args)

        checkpoint = torch.load(self.args.resume)

        prefix = "module."
        sd = {
            name[len(prefix):]: value for name, value in checkpoint['state_dict'].items()
        }

        self.model.load_state_dict(sd, strict=True)
        self.model = torch.nn.DataParallel(self.model).cuda()

        self.eval_w = 1248
        self.eval_h = 384

    def load(self, left, right):
        size = np.shape(left)
        height = size[0]
        width = size[1]
        temp_data = np.zeros([6, height, width], np.float32)
        left = np.asarray(left)
        right = np.asarray(right)
        r = left[:, :, 0]
        g = left[:, :, 1]
        b = left[:, :, 2]
        temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
        temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
        temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
        r = right[:, :, 0]
        g = right[:, :, 1]
        b = right[:, :, 2]
        # r,g,b,_ = right.split()
        temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
        temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
        temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
        return temp_data

    def transform(self, temp_data):
        _, h, w = np.shape(temp_data)

        if h <= self.eval_h and w <= self.eval_w:
            # padding zero
            temp = temp_data
            temp_data = np.zeros([6, self.eval_h, self.eval_w], np.float32)
            temp_data[:, self.eval_h - h: self.eval_h, self.eval_w - w: self.eval_w] = temp
        else:
            start_x = int((w - self.eval_w) / 2)
            start_y = int((h - self.eval_h) / 2)
            temp_data = temp_data[:, start_y: start_y + self.eval_h, start_x: start_x + self.eval_w]
        left = np.ones([1, 3, self.eval_h, self.eval_w], np.float32)
        left[0, :, :, :] = temp_data[0: 3, :, :]
        right = np.ones([1, 3, self.eval_h, self.eval_w], np.float32)
        right[0, :, :, :] = temp_data[3: 6, :, :]
        return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

    def disparity(self, frame: KittiFrame):
        input1, input2, height, width = self.transform(self.load(frame.left_color(), frame.right_color()))

        input1 = Variable(input1, requires_grad=False)
        input2 = Variable(input2, requires_grad=False)

        self.model.eval()
        input1 = input1.cuda()
        input2 = input2.cuda()
        torch.cuda.synchronize()
        with torch.no_grad():
            prediction = self.model(input1, input2)
        torch.cuda.synchronize()

        temp = prediction.cpu()
        temp = temp.detach().numpy()
        if height <= self.eval_h or width <= self.eval_w:
            temp = temp[0, self.eval_h - height: self.eval_h, self.eval_w - width: self.eval_w]
        else:
            temp = temp[0, :, :]

        return temp
