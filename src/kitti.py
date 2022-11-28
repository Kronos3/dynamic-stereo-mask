import cv2

from pathlib import Path


class KittiDataset:
    def __init__(self, basepath: Path) -> None:
        self.basepath = basepath

        self.calib = basepath / 'calib.txt'

    def left(self, index: int) -> Path:
        return self.basepath / 'image_2' / ("%06d.png" % index)

    def right(self, index: int) -> Path:
        return self.basepath / 'image_3' / ("%06d.png" % index)


class KittiFrame:
    def __init__(self, dataset: KittiDataset, index: int) -> None:
        self._dataset = dataset
        self.index = index

        self._left_color = None
        self._right_color = None
        self._left_gray = None
        self._right_gray = None

    def left_color(self) -> cv2.Mat:
        if self._left_color is None:
            self._left_color = cv2.imread(str(self._dataset.left(self.index)))
            self._left_color = cv2.cvtColor(self._left_color, cv2.COLOR_BGR2RGB)
        return self._left_color

    def right_color(self) -> cv2.Mat:
        if self._right_color is None:
            self._right_color = cv2.imread(str(self._dataset.right(self.index)))
            self._right_color = cv2.cvtColor(self._right_color, cv2.COLOR_BGR2RGB)
        return self._right_color

    def left_gray(self) -> cv2.Mat:
        if self._left_gray is None:
            self._left_gray = cv2.cvtColor(self.left_color(), cv2.COLOR_RGB2GRAY)
        return self._left_gray

    def right_gray(self) -> cv2.Mat:
        if self._right_gray is None:
            self._right_gray = cv2.cvtColor(self.right_color(), cv2.COLOR_RGB2GRAY)
        return self._right_gray
