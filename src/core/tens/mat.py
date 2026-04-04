import numpy as np
import core.tens.vec as v

class Matrix:
    def __init__(self, *args,shape=None):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            self.data = args[0].astype(float)
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            arr = np.array(args[0], dtype=float)
            if arr.ndim == 1:
                if shape is None:
                    raise ValueError("Invalid shape")
                self.data = arr.reshape(shape)
            else:
                self.data = arr
        else:
            arr = np.array(args, dtype=float)

            if arr.ndim == 1:
                if shape is None:
                    raise ValueError("Invalid shape")
                self.data = arr.reshape(shape)
            else:
                self.data = arr
        if self.data.ndim != 2:
            raise ValueError("Invalid ndim")
        self.n, self.m = self.data.shape
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return self.__class__((self.data @ other.data))
        elif isinstance(other, v.Vector):
            result = self.data @ other.data
            return v.Vector(*result)
        raise TypeError

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data}\n)"
    def transpose(self):
        return self.__class__(self.data.T)
    @property
    def T(self):
        return self.transpose()
    def inverse(self):
        return self.__class__(np.linalg.inv(self.data))

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Matrix(self.data * scalar)
        raise TypeError
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data + other.data)
        raise TypeError
    @staticmethod
    def identity(size):
        return Matrix(np.eye(size))
    @staticmethod
    def zeros(n, m):
        return Matrix(np.zeros((n, m)))
    @staticmethod
    def ones(n, m):
        return Matrix(np.ones((n, m)))
    @property
    def shape(self):
        return self.data.shape
    def copy(self):
        return Matrix(self.data.copy())
    def is_sq(self):
        return self.n==self.m
    def inverse(self):
        if not self.is_sq():
            raise ValueError
        return Matrix(np.linalg.inv(self.data))
    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, value):
        self.data[key] = value
    def determinant(self):
        if not self.is_sq():
            raise ValueError
        return float(np.linalg.det(self.data))
    def is_upper_tri(self,eps):
        n = len(self.data)
        for i in range(0,n):
            for j in range(0,i):
                if np.abs(self[i][j])>eps:
                    return False
        return True
    def get_col(self, j):
        return v.Vector(*self.data[:,j])
    def get_row(self, i):
        return v.Vector(*self.data[i,:])
