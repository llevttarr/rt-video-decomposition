from ..core.tens.vec import Vector
from ..core.tens.mat import Matrix

class FrameBuffer:
    def __init__(self,n:int):
        self.n=n
        self.frames:list[Vector]=[]
    def push(self,frame:Vector):
        # if (self.is_full):
        pass
    def is_full(self):
        return len(self.frames)>=self.n
    def to_mat(self)->Matrix:
        pass
