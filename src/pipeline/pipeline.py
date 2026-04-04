from ..core.tens.vec import Vector
from ..core.tens.mat import Matrix

from ..io.input import get_frame

from frame_buffer import FrameBuffer
from results import PipelineResult
from background_model import BackgroundModel
class VideoPipeline:
    def __init__(self,n:int):
        self.buffer=FrameBuffer(n)

    def process(self):
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
        v:Vector=get_frame()
        self.preprocess(v)

        res= PipelineResult(v)

        buffer=self.buffer
        if (buffer.is_full()):
            model=BackgroundModel(rank=0,threshold=0.0)
            x:Matrix=buffer.to_mat()
            model.process(x,v)

            res.background=model.bg
            res.foreground_mask=model.fg
        buffer.push(v)
        res.output()
    def preprocess(self,v:Vector):
        pass
