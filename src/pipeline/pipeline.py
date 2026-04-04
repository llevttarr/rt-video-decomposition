from core.tens.vec import Vector
from core.tens.mat import Matrix

from debug.debug import dbg

from pipeline.frame_buffer import FrameBuffer
from pipeline.results import PipelineResult
from pipeline.background_model import BackgroundModel

import numpy as np
import cv2

WIDTH_RESIZE=90
HEIGHT_RESIZE=90
RANK=2
THRESHOLD=10.0

class VideoPipeline:
    def __init__(self,n:int,w:int,h:int):
        dbg("pipeline init")
        self.buffer=FrameBuffer(n)
        self.model=BackgroundModel(rank=RANK,threshold=THRESHOLD)
        self.w=w
        self.h=h

    def process(self,frame):
        # STAGES
        # - - -
        # 1. get raw frame
        # 2. preprocess
        # 2.1. grayscale 
        # 2.2. resize 
        # 2.3. flatten 
        # 3. if buffer large enough -> process
        # 3.1. compute background estimation
        # 3.2. compute foreground
        # 3.3. recompute result
        # 4. push into buffer
        # 5. output result
        # - - -
        v,vshape=self.preprocess(frame)

        res= PipelineResult(frame,v,vshape,None,None)

        buffer=self.buffer
        if (buffer.is_full()):
            x:Matrix=buffer.to_mat()
            self.model.process(x,v)

            res.background=self.model.bg
            res.foreground_mask=self.model.fg
        buffer.push(v)
        return res.output(self.w,self.h)
    def preprocess(self,frame)->tuple[Vector,tuple[int,int]]:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        target_width=WIDTH_RESIZE
        target_height=HEIGHT_RESIZE
        gray = cv2.resize(gray,(target_width, target_height), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float64)
        flat = gray.flatten()
        vec = Vector(flat.tolist())
        return vec, gray.shape
