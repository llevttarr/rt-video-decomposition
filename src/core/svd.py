import core.tens.mat as matu
import core.tens.vec as vecu
import core.eigen as eig

import numpy as np
def get_U(A:matu.Matrix,eigval,eigvec):
    m,n = A.shape
    U = matu.Matrix.zeros(m, m)
    for i in range(min(m, n)):
        sigma_i = sqrt_safe(eigval[i])
        if sigma_i>1e-10:
            vi = eigvec[i].normalize()
            ui = (A@vi)/sigma_i
            ui = ui.normalize()
        else:
            ui =vecu.Vector(*([0.0] * m))
        for k in range(m):
            U[k,i] = ui[k]
    return U
def get_sigma(A:matu.Matrix,eigval):
    m,n = A.shape
    S = matu.Matrix.zeros(m, n)
    for i in range(min(m, n)):
        S[i,i]=sqrt_safe(eigval[i])
    return S
def get_V(A:matu.Matrix,eigvec):
    n = A.shape[1]
    V = matu.Matrix.zeros(n, n)
    for j, vec in enumerate(eigvec):
        normalized = vec.normalize()
        for i in range(n):
            V[i,j] = normalized[i]
    return V

def sort_eigval(eigval,eigvec):
    paired = sorted(zip(eigval,eigvec), key=lambda x:x[0], reverse=True)
    sorted_vals = [p[0] for p in paired]
    sorted_vecs = [p[1] for p in paired]
    return sorted_vals, sorted_vecs

def sqrt_safe(x):
    return (max(x,0.0))**0.5
def svd_decomposition(A:matu.Matrix):
    AtA = A.T @ A
    eigval,eigvec= eig.get_eig(AtA)
    eigval,eigvec=sort_eigval(eigval,eigvec)

    U=get_U(A,eigval,eigvec)
    S=get_sigma(A,eigval)
    V=get_V(A,eigvec)
    return U,S,V
