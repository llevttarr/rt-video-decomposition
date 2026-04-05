from core.tens.vec import Vector
from core.tens.mat import Matrix

from core.svd import svd_decomposition
from core.qr import qr_decomposition

from debug.debug import dbg

import numpy as np

class BackgroundModel():
    def __init__(self,rank:int,threshold:float):
        self.rank=rank
        self.threshold=threshold
        self.bg=None
        self.fg=None
    def process(self,x:Matrix,v:Vector):
        self.bg=self.estimate_background(x,v)
        self.fg=self.get_foreground(v)
    def estimate_background(self,x,framev):
        u,s,v=svd_decomposition(x)
        r=min(self.rank,u.shape[1])
        if r<=0:
            dbg("rank <= 0")
            raise ValueError()
        ur= Matrix(u.data[:, :r])
        coeff=ur.T @ framev
        bg=ur@coeff
        return bg
    def get_foreground(self,v:Vector):
        residual = np.abs(v.data-self.bg.data)
        mask = np.where(residual>self.threshold,255.0,0.0)
        return Vector(*mask)
