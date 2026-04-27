from core.tens.vec import Vector
from core.tens.mat import Matrix

from debug.debug import dbg

from pipeline.frame_buffer import FrameBuffer
from pipeline.results import PipelineResult
from pipeline.background_model import BackgroundModel

import numpy as np
import cv2

# optimised for CUDA, reduce if bad performance
WIDTH_RESIZE=480
HEIGHT_RESIZE=270
RANK=4
THRESHOLD=30.0

class VideoPipeline:
    def __init__(self,n:int,w:int,h:int,use_cuda):
        dbg("pipeline init")
        self.buffer=FrameBuffer(n)
        self.model=BackgroundModel(rank=RANK,threshold=THRESHOLD,use_cuda=use_cuda)
        self.color_mode=self.model.use_cuda
        self.w=w
        self.h=h

    def process(self,frame):
        # - - -
        # - - -
        v,vshape,channels=self.preprocess(frame)

        res= PipelineResult(frame,v,vshape,None,None)

        
        if not self.model.initialized:
            self.buffer.push(v)
            if self.buffer.is_full():
                x=self.buffer.to_mat().data
                self.model.init_model(x,spatial_shape=vshape,channels=channels)
            return res.output(self.w, self.h)

        self.model.process(v)
        res.background=self.model.bg
        # res.foreground_mask=self.model.fg
        res.foreground_mask = self.postprocess_mask(self.model.fg, vshape)
        return res.output(self.w,self.h)
    def preprocess(self,frame)->tuple[Vector,tuple[int,int],int]:
        target_width=WIDTH_RESIZE
        target_height=HEIGHT_RESIZE
        frame_small = cv2.resize(frame,(target_width, target_height), interpolation=cv2.INTER_AREA)

        if self.color_mode:
            color = frame_small.astype(np.float32, copy=False)
            vec = Vector.from_array(color.reshape(-1))
            return vec, (target_height, target_width), 3

        gray = cv2.cvtColor(frame_small,cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32, copy=False)
        vec = Vector.from_array(gray.reshape(-1))
        return vec, gray.shape, 1
    def postprocess_mask(self, fg: Vector, shape) -> Vector:
        mask = fg.data.reshape(shape).astype(np.uint8)
        kernel_open = np.ones((3,3), np.uint8)
        kernel_close = np.ones((5,5), np.uint8)
        mask= cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel_open)
        mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask=cv2.dilate(mask,kernel_open,iterations=1)

        return Vector.from_array(mask.reshape(-1))
