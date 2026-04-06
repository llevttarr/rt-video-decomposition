from core.tens.vec import Vector
from core.tens.mat import Matrix

from core.svd import svd_decomposition
from core.qr import qr_decomposition


from debug.debug import dbg

import numpy as np

class BackgroundModel():
    def __init__(self,rank:int,threshold:float,rate=0.001,use_cuda=False):
        self.rank=rank
        self.threshold=threshold
        self.bg=None
        self.fg=None
        self.mean=None
        self.u=None
        self.rate=rate
        self.initialized=False

        self._qr_fn = None
        self._svd_fn = None
        if use_cuda:
            try:
                from gpu.cuda_alg import qr_gpu
                from gpu.cuda_alg import svd_gpu
            except ImportError as exc:
                dbg(f"CUDA modules unavailable, fallback to CPU: {exc}")
                use_cuda = False
            else:
                self._qr_fn = qr_gpu
                self._svd_fn = svd_gpu
        self.use_cuda=use_cuda

    def init_model(self,x:Matrix):
        x_data = x.data if isinstance(x, Matrix) else np.asarray(x, dtype=float)
        self.mean=np.mean(x_data,axis=1)
        xmd=x_data-self.mean[:,None]
        if self.use_cuda:
            try:
                u,_,_=self._svd_fn(xmd)
            except Exception as exc:
                dbg(f"CUDA SVD unavailable, fallback to CPU: {exc}")
                self.use_cuda = False
                u,_,_=svd_decomposition(Matrix(xmd))
        else:
            u,_,_=svd_decomposition(Matrix(xmd))
        u_data = u.data if isinstance(u, Matrix) else np.asarray(u, dtype=float)
        self.initialized=True

        r=min(u_data.shape[1],self.rank)
        self.u=u_data[:,:r]
        dbg("model init")
        
    def process(self,v:Vector):
        x=v.data
        z=x-self.mean

        coeff=self.u.T @ z
        z_bg=self.u @ coeff
        bg=self.mean+z_bg
        self.bg=Vector(*bg)
        self.fg=self.get_foreground(v)
        self.upd_model_masked(x)
    def get_foreground(self,v:Vector):
        residual = np.abs(v.data-self.bg.data)
        mask = np.where(residual>self.threshold,255.0,0.0)
        return Vector(*mask)
    def upd_model(self,x,z,eps=1e-6):
        rate=self.rate
        self.mean=self.mean*(1.0-rate)+rate*x
        coeff=self.u.T @ z
        z_proj=self.u @ coeff
        residual=z-z_proj
        norm=np.linalg.norm(residual)
        if norm >eps:
            new_d=residual/norm
            u_updat=np.column_stack([self.u,new_d])
            u_upd=Matrix(u_updat)
            if self.use_cuda:
                try:
                    q,_=self._qr_fn(u_updat)
                except Exception as exc:
                    dbg(f"CUDA QR unavailable, fallback to CPU: {exc}")
                    self.use_cuda = False
                    q,_=qr_decomposition(u_upd)
            else:
                q,_=qr_decomposition(u_upd)
            q_data = q.data if isinstance(q, Matrix) else np.asarray(q, dtype=float)
            self.u=q_data[:,:self.rank]
    def upd_model_masked(self,x,eps=1e-6):
        rate = self.rate
        fg_mask = self.fg.data > 0
        bg_mask = ~fg_mask
        if np.sum(bg_mask) == 0:
            return
        self.mean[bg_mask] = self.mean[bg_mask]*(1.0-rate)+rate*x[bg_mask]
        z = x-self.mean
        coeff = self.u.T @ z
        z_proj = self.u @ coeff
        residual = z-z_proj
        residual[fg_mask] = 0.0
        norm = np.linalg.norm(residual)
        if norm > eps:
            new_d = residual / norm
            u_updat = np.column_stack([self.u, new_d])
            u_upd=Matrix(u_updat)
            if self.use_cuda:
                try:
                    q,_=self._qr_fn(u_updat)
                except Exception as exc:
                    dbg(f"CUDA QR unavailable, fallback to CPU: {exc}")
                    self.use_cuda = False
                    q,_=qr_decomposition(u_upd)
            else:
                q,_=qr_decomposition(u_upd)
            q_data = q.data if isinstance(q, Matrix) else np.asarray(q, dtype=float)
            self.u = q_data[:, :self.rank]
