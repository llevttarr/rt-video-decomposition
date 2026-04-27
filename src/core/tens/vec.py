import numpy as np

class Vector:
    def __init__(self, *args):
        self.size=len(args)
        self.data = np.array(args, dtype=float)

    @classmethod
    def from_array(cls, arr):
        obj = cls.__new__(cls)
        data = np.asarray(arr, dtype=float).reshape(-1)
        obj.size = data.size
        obj.data = data
        return obj

    def __add__(self, other):
        return Vector(*(self.data + other.data))

    def __sub__(self, other):
        return Vector(*(self.data - other.data))

    def __mul__(self, scalar): 
        return Vector(*(self.data * scalar))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __matmul__(self, other):
        return np.dot(self.data, other.data)

    def __getitem__(self, index):
        return self.data[index]

    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    def __repr__(self):
        return f"Vector{len(self.data)}D({', '.join(map(str, self.data))})"
    @property
    def length(self):
        return np.linalg.norm(self.data)
    def __truediv__(self,c):
        return Vector(*(self.data/c))

    def normalize(self):
        norm = self.length
        return self.__class__(*(self.data / norm)) if norm != 0 else self

class Vector2D(Vector):
    def __init__(self, vx, vy):
        super().__init__(vx, vy)

class Vector3D(Vector):
    def __init__(self, vx, vy, vz):
        super().__init__(vx, vy, vz)

class Vector4D(Vector):
    def __init__(self, x, y, z, w):
        super().__init__(x, y, z, w)
