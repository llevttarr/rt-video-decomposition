from dataclasses import dataclass
from core.tens.vec import Vector
from core.tens.mat import Matrix

import numpy as np
import cv2

BG_COLOR=(255,255,255)

@dataclass
class PipelineResult:
    frame: np.ndarray
    processed: Vector
    vshape:tuple[int,int]
    background: Vector| None
    foreground_mask: Vector| None
    def vector_to_frame(self, v: Vector,w,h)-> np.ndarray:
        arr = np.array(v.data, dtype=np.float64)
        arr =arr.reshape((h,w))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    def output(self,w,h):
        if self.foreground_mask is None:
            return self.frame

        h,w = self.vshape
        mask_small = self.vector_to_frame(self.foreground_mask, w, h)
        frame_h, frame_w = self.frame.shape[:2]
        mask = cv2.resize(mask_small, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        fg_bool = mask.astype(bool)
        out = np.full_like(self.frame, BG_COLOR, dtype=np.uint8)
        out[fg_bool] = self.frame[fg_bool]

        return out
