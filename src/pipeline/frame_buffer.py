from core.tens.vec import Vector
from core.tens.mat import Matrix

import numpy as np

class FrameBuffer:
    def __init__(self,n:int):
        self.n=n
        self.frames:list[Vector]=[]
    def push(self,frame:Vector):
        if (not self.is_full()):
            self.frames.append(frame)
            return False

        for i in range(len(self.frames)-1):
            self.frames[i]=self.frames[i+1]
        self.frames[self.n-1]=frame
        return True
    def is_full(self):
        return len(self.frames)>=self.n
    def to_mat(self)->Matrix:
        fr = [v.data for v in self.frames]
        return Matrix(np.column_stack(fr))
