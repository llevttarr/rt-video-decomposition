import core.tens.mat as matu
import core.tens.vec as vecu
import core.qr

def get_eig(A: matu.Matrix, max_iter=1000, eps=1e-10):
    if not A.is_sq():
        raise ValueError()

    n = A.shape[0]
    Ak = A.copy()
    Q_total = matu.Matrix.identity(n)

    for _ in range(max_iter):
        Q, R = qr.qr_decomposition(Ak)
        Ak = R @ Q
        Q_total = Q_total @ Q
        if Ak.is_upper_tri(eps):
            break
    eigval = [Ak[i, i] for i in range(n)]
    eigvec = [Q_total.get_col(i) for i in range(n)]

    return eigval, eigvec


def get_eigval(A: matu.Matrix, max_iter=1000, eps=1e-10):
    if not A.is_sq():
        raise ValueError()
    n = A.shape[0]
    Ak = A.copy()
    for _ in range(max_iter):
        Q, R = qr.qr_decomposition(Ak)
        Ak = R @ Q
        if Ak.is_upper_tri(eps):
            break
    return [Ak[i,i] for i in range(n)]

def get_eigvec(A: matu.Matrix, max_iter=1000, eps=1e-10):
    _,eigvec = get_eig(A, max_iter, eps)
    return eigvec