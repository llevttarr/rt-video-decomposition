import numpy as np
import tens.mat as matu
import tens.vec as vecu

def qr_decomposition(A:matu.Matrix):
    n,m= A.shape
    Q=matu.Matrix.identity(n)
    R=A.copy()
    
    for i in range(min(n-1, m)):
        col = R.get_col(i)
        x = vecu.Vector(*[col[j] for j in range(i, n)])
        e1 = vecu.Vector(*([1.0]+[0.0] * (len(x.data) - 1)))
        sign = 1.0 if x[0] >= 0 else -1.0
        u = x + sign*x.length*e1

        if u.length == 0:
            continue

        u = u.normalize()

        u_col = u.data.reshape(-1,1)
        H_small = np.eye(n-i) - 2.0 * (u_col @ u_col.T)

        H = np.eye(n)
        H[i:,i:] = H_small
        H = matu.Matrix(H)

        R = H@R
        Q = Q@H
    return Q,R
