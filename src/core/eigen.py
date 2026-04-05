import core.tens.mat as matu
import core.tens.vec as vecu
import core.qr as qr

import numpy as np

def get_eig(A: matu.Matrix, max_iter=1000, eps=1e-10):
    if not A.is_sq():
        raise ValueError()

    n = A.shape[0]
    Ak = A.copy()
    Q_total = matu.Matrix.identity(n)
    I = matu.Matrix.identity(n)

    for iteration in range(max_iter):
        shift_val = Ak[n-1, n-1]
        shift_mat = I * shift_val
        Q, R = qr.qr_decomposition(Ak - shift_mat)
        Ak = (R @ Q) + shift_mat
        Q_total = Q_total @ Q
        if iteration % 5 == 0:
            lower_tri = np.tril(Ak.data, k=-1)
            if np.max(np.abs(lower_tri)) < eps:
                break
    eigval = [Ak[i, i] for i in range(n)]
    eigvec = [vecu.Vector(*Q_total.data[:, i]) for i in range(n)]

    return eigval, eigvec


def get_eigval(A: matu.Matrix, max_iter=1000, eps=1e-10):
    eigval,_ = get_eig(A, max_iter, eps)
    return eigval

def get_eigvec(A: matu.Matrix, max_iter=1000, eps=1e-10):
    _,eigvec = get_eig(A, max_iter, eps)
    return eigvec