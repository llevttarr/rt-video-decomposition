from core.tens.vec import Vector
from core.tens.mat import Matrix

import cv2
import numpy as np
import sys

from debug.debug import dbg

class CameraInput:
    def __init__(self,index,w,h,fps):
        dbg("camerainput init")
        self.index=index
        self.w=w
        self.h=h
        self.fps=fps
        self.cap = None
    def capture(self) -> None:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        if not self.cap.isOpened():
            dbg("camerainput capture() break")
            raise RuntimeError(f"err: camera not opened {self.index}")
    def get_frame(self) -> np.ndarray | None:
        if self.cap is None:
            dbg("camerainput get_frame() break")
            raise RuntimeError("err: camera not opened")
        res,frame = self.cap.read()
        if not res:
            return None
        return frame
    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
