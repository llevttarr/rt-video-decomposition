import numpy as np
import vec as v

class Matrix:
    def __init__(self, *args):
        self.data = np.array(args, dtype = float).reshape((self.size, self.size))
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return self.__class__((self.data @ other.data).flatten())
        elif isinstance(other, v.Vector):
            result = self.data @ other.data
            return v.Vector(*result)
        raise TypeError

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data}\n)"

    def transpose(self):
        return self.__class__(self.data.T.flatten())
    def inverse(self):
        return self.__class__(np.linalg.inv(self.data))
class Matrix2D(Matrix):
    size = 2
    def __init__(self, *args):
        super().__init__(args)
class Matrix3D(Matrix):
    size = 3
    def __init__(self, *args):
        super().__init__(args)

class Matrix4D(Matrix):
    size = 4
    def __init__(self, *args):
        super().__init__(args)
