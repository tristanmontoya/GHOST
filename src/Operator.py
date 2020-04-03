# GHOST - Operator Template Language

import numpy as np


class Operator:
    """Generic operator class"""

    def __init__(self, shape_in, shape_out, domain=lambda x: True):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.domain = domain  # returns True if x is in domain false if not.
        self.vec = None
        self.mat = None

    def __add__(self, other):
        if self.shape_in != other.shape_in or self.shape_out != other.shape_out:
            raise ValueError("Incompatible operations for addition")

        if self.vec is not None and other.vec is not None:  # both diagonal
            return DiagonalOperator(self.vec + other.vec)
        if self.vec is not None and other.mat is not None:  # pre-multiply by diagonal
            return DenseLinearOperator(np.diag(self.vec) + other.mat)
        if self.mat is not None and other.vec is not None:  # post-multiply by diagonal
            return DenseLinearOperator(self.mat + np.diag(other.vec))
        if self.mat is not None and other.mat is not None:  # both dense
            return DenseLinearOperator(self.mat + other.mat)
        else:
            new = Operator(self.shape_in, self.shape_out)
            new.function = lambda ar: self.function(ar) + other.function(ar)
            return new

    def __sub__(self,other):
        return self + -1.0*other

    def __mul__(self, innerOperator):

        # check for scalar
        if np.isscalar(innerOperator):
            if self.vec is not None:
                return DiagonalOperator(self.vec * innerOperator)
            if self.mat is not None:
                return DenseLinearOperator(self.mat * innerOperator)
            else:
                new = Operator(self.shape_in, self.shape_out)
                new.function = lambda arg: innerOperator * self.function(arg)
                return new

        # check compatibility
        if self.shape_in != innerOperator.shape_out:
            raise ValueError("Domain of inner function must match "
                             "codomain of outer function for composition.")

        # check for identity
        if innerOperator.__class__ == Identity:
            return self
        if self.__class__ == Identity:
            return innerOperator

        # check if NumPy can be used
        if self.vec is not None and innerOperator.vec is not None: # both diagonal
            return DiagonalOperator(self.vec*innerOperator.vec)
        if self.vec is not None and innerOperator.mat is not None: # pre-multiply by diagonal
            return DenseLinearOperator(np.einsum("i,ij->ij", self.vec, innerOperator.mat))
        if self.mat is not None and innerOperator.vec is not None: # post-multiply by diagonal
            return DenseLinearOperator(np.einsum("ij,j->ij", self.mat,innerOperator.vec))
        if self.mat is not None and innerOperator.mat is not None: # both dense
            return DenseLinearOperator(self.mat @ innerOperator.mat)
        else:
            new = Operator(innerOperator.shape_in, self.shape_out)
            new.function = lambda arg: self.function(innerOperator.function(arg))
            return new

    def __rmul__(self, scalar):

        # pre-multiply by scalar
        if not np.isscalar(scalar):
            raise TypeError
        if self.vec is not None:
            return DiagonalOperator(scalar*self.vec)
        if self.mat is not None:
            return DenseLinearOperator(scalar*self.mat)
        else:
            new = Operator(self.shape_in, self.shape_out)
            new.function = lambda arg: scalar * self.function(arg)
            return new

    def __pow__(self, power, modulo=None):
        if self.shape_in != self.shape_out:
            raise TypeError("Operator must be an endomorphism to exponentiate.")
        if int(power) != power:
            raise TypeError("Operator must be taken to integer power.")
        if power == 0:
            return Identity(self.shape_in)
        if power == 1:
            return self
        return self*self**(power-1)

    def function(self, arg: np.ndarray):
        raise NotImplementedError

    def __call__(self, arg: np.ndarray):
        if not self.domain(arg):
            raise ValueError("Operator input must be within domain.")
        if not arg.shape == self.shape_in:
            raise ValueError("Incompatible input dimension")

        return self.function(arg)


class DenseLinearOperator(Operator):
    """Dense matrix operator on column vectors"""

    def __init__(self, mat: np.ndarray) -> None:
        assert(len(mat.shape) == 2), "DenseLinearOperator must be initialized with 2D array."
        shape_in = (mat.shape[1],)
        shape_out = (mat.shape[0],)

        super().__init__(shape_in, shape_out)
        self.mat = mat

        if shape_in == shape_out:
            if np.allclose(np.diag(np.diag(self.mat)), self.mat):
                self.vec = np.diag(self.mat)
                self.mat = None
                if np.allclose(np.ones(self.shape_in), self.vec):
                    self.__class__ = Identity
                else:
                    self.__class__ = DiagonalOperator

    def __repr__(self):
        return str(self.mat)

    @property
    def inv(self):
            return DenseLinearOperator(np.linalg.inv(self.mat))

    @property
    def T(self):
        return DenseLinearOperator(self.mat.T)

    def is_close(self, other):
        return np.allclose(self.mat, other.mat)

    def function(self, arg):
        return self.mat @ arg


class DiagonalOperator(Operator):
    def __init__(self, vec: np.ndarray) -> None:
        assert (len(vec.shape) == 1), "Diagonal Operator must be initialized with 1D array."
        shape_in = (vec.shape[0],)
        shape_out = (vec.shape[0],)

        super().__init__(shape_in, shape_out)
        self.vec = vec
        self.mat = np.diag(vec)

        if shape_in == shape_out and np.allclose(np.ones(self.shape_in), self.vec):
            self.__class__ = Identity

    def __repr__(self):
        return 'Diagonal Matrix: ' + str(self.vec)

    @property
    def inv(self):
            return DiagonalOperator(1./self.vec)

    @property
    def T(self):
        return self

    def is_close(self, other):
        return np.allclose(self.vec, other.vec)

    def function(self, arg):
        return np.einsum("i, i -> i", self.vec, arg)


class Identity(Operator):

    def __init__(self, N: int) -> None:
        super().__init__((N,), (N,))
        self.vec = np.ones(N)

    @property
    def inv(self):
        return self

    @property
    def T(self):
        return self

    def __repr__(self):
        return 'Identity of size ' + str(self.shape_in[0])

    def function(self, arg):
        return arg



