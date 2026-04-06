import ctypes
import pathlib
import sysconfig


def _preload_cuda_wheel_libs() -> None:
    base = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    candidates = [
        base / "cuda_runtime/lib/libcudart.so.12",
        base / "cuda_nvrtc/lib/libnvrtc.so.12",
        base / "curand/lib/libcurand.so.10",
        base / "cublas/lib/libcublas.so.12",
        base / "cusparse/lib/libcusparse.so.12",
        base / "cusolver/lib/libcusolver.so.11",
        base / "nvjitlink/lib/libnvJitLink.so.12",
    ]
    for lib_path in candidates:
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


_preload_cuda_wheel_libs()

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
