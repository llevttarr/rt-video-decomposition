import numpy as np
import core.tens.mat as matu

def qr_decomposition(A:matu.Matrix):
    n,m= A.shape
    Q_data=np.eye(n)
    R_data=A.data.copy()
    
    for i in range(min(n - 1, m)):
        x = R_data[i:, i]
        
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        
        sign = 1.0 if x[0] >= 0 else -1.0
        u = x + sign * norm_x * e1
        
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            continue
            
        u = u / norm_u
        u_col = u.reshape(-1, 1)
        R_sub = R_data[i:, i:]
        R_data[i:, i:] = R_sub - 2.0 * u_col @ (u_col.T @ R_sub)
        Q_sub = Q_data[:, i:]
        Q_data[:, i:] = Q_sub - 2.0 * (Q_sub @ u_col) @ u_col.T
    return matu.Matrix(Q_data), matu.Matrix(R_data)
