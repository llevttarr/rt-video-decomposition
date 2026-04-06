import cupy
import cupy.linalg as cupyl 
import core.tens.mat as matu

def qr_gpu(an):
    a=cupy.asarray(an)
    q,r=cupyl.qr(a)
    nq,nr=cupy.asnumpy(q), cupy.asnumpy(r)
    return matu.Matrix(nq),matu.Matrix(nr)
def svd_gpu(an):
    a=cupy.asarray(an)
    u,s,v=cupyl.svd(a,full_matrices=False)
    nu,ns,nv=cupy.asnumpy(u),cupy.asnumpy(cupy.diag(s)),cupy.asnumpy(v)
    return matu.Matrix(nu),matu.Matrix(ns),matu.Matrix(nv)
