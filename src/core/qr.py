import numpy as np
import core.tens.mat as matu

def qr_decomposition(A:matu.Matrix):
    n,m= A.shape
    k = min(n, m)
    Q_data=np.eye(n, k)
    R_data=A.data.copy()
    reflectors: list[tuple[int, np.ndarray]] = []
    
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
        reflectors.append((i, u_col))

    for i, u_col in reversed(reflectors):
        Q_sub = Q_data[i:, :]
        Q_data[i:, :] = Q_sub - 2.0 * u_col @ (u_col.T @ Q_sub)

    return matu.Matrix(Q_data), matu.Matrix(R_data[:k, :])
