import tens.mat as matu
import tens.vec as vecu
import eigen as eig
import numpy as np
def get_U(A:matu.Matrix):
    return
def get_sigma(A:matu.Matrix):
    updA=A
    # todo
    eigenval=eig.get_eigenvalues(updA)
    eigenval = np.sqrt(eigenval)
    return eigenval[::-1] 
def get_V(A:matu.Matrix):
    return
def svd_decomposition(A:matu.Matrix):
    U=get_U(A)
    S=get_sigma(A)
    V=get_V(A)
    return U,S,V
