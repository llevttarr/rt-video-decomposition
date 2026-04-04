from ..core.tens.vec import Vector
from ..core.tens.mat import Matrix

class BackgroundModel():
    def __init__(self,rank:int,threshold:float):
        self.rank=rank
        self.threshold=threshold
        self.bg=None
        self.fg=None
    def process(self,X:Matrix,frame:Vector):
        pass
    def estimate_background(self):
        pass
    def get_foreground(self,frame:Vector):
        pass
