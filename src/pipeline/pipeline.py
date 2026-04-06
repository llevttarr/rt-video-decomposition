from core.tens.vec import Vector
from core.tens.mat import Matrix

from debug.debug import dbg

from pipeline.frame_buffer import FrameBuffer
from pipeline.results import PipelineResult
from pipeline.background_model import BackgroundModel

import numpy as np
import cv2

WIDTH_RESIZE=320
HEIGHT_RESIZE=240
RANK=3
THRESHOLD=25.0

class VideoPipeline:
    def __init__(self,n:int,w:int,h:int,use_cuda):
        dbg("pipeline init")
        self.buffer=FrameBuffer(n)
        self.model=BackgroundModel(rank=RANK,threshold=THRESHOLD,use_cuda=use_cuda)
        self.w=w
        self.h=h

    def process(self,frame):
        # - - -
        # - - -
        v,vshape=self.preprocess(frame)

        res= PipelineResult(frame,v,vshape,None,None)

        
        if not self.model.initialized:
            self.buffer.push(v)
            if self.buffer.is_full():
                x=self.buffer.to_mat().data
                self.model.init_model(x)
            return res.output(self.w, self.h)

        self.model.process(v)
        res.background=self.model.bg
        # res.foreground_mask=self.model.fg
        res.foreground_mask = self.postprocess_mask(self.model.fg, vshape)
        return res.output(self.w,self.h)
    def preprocess(self,frame)->tuple[Vector,tuple[int,int]]:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        target_width=WIDTH_RESIZE
        target_height=HEIGHT_RESIZE
        gray = cv2.resize(gray,(target_width, target_height), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32)
        flat = gray.flatten()
        vec = Vector(*flat.tolist())
        return vec, gray.shape
    def postprocess_mask(self, fg: Vector, shape) -> Vector:
        mask = fg.data.reshape(shape).astype(np.uint8)
        kernel_open = np.ones((3,3), np.uint8)
        kernel_close = np.ones((5,5), np.uint8)
        mask= cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel_open)
        mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask=cv2.dilate(mask,kernel_open,iterations=1)

        return Vector(*mask.flatten().tolist())
