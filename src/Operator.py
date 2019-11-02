# In development

import numpy as np


class Operator:
    """Generic operator class"""

    def __init__(self, shape_in, shape_out, domain=lambda x: True):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.domain = domain  # returns True if x is in domain false if not.
        self.mat = None

    def __add__(self, other):
        if self.shape_in != other.shape_in or self.shape_out != other.shape_out:
            raise ValueError("Incompatible operations for addition")

        try:
            return DenseLinearOperator(self.mat + other.mat)
        except AttributeError:
            new = Operator(self.shape_in, self.shape_out)
            new.function = lambda ar: self.function(ar) + other.function(ar)
            return new

    def __sub__(self,other):
        return self + -1.0*other

    def __mul__(self, innerOperator):

        # check for scalar
        if np.isscalar(innerOperator):
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
        if self.mat is not None:
            return DenseLinearOperator(self.mat @ innerOperator.mat)
        else:
            new = Operator(innerOperator.shape_in, self.shape_out)
            new.function = lambda arg: self.function(innerOperator.function(arg))
            return new

    def __rmul__(self, scalar):

        # pre-multiply by scalar
        if not np.isscalar(scalar):
            raise TypeError
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
        shape_in = (mat.shape[1], 1)
        shape_out = (mat.shape[0], 1)

        super().__init__(shape_in, shape_out)
        self.mat = mat

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


class Identity(Operator):

    def __init__(self, N: int) -> None:
        super().__init__((N, 1), (N,1))
        self.mat = np.eye(N)

    @property
    def inv(self):
        return self

    @property
    def T(self):
        return self

    def function(self, arg):
        # this is what makes it different than just a matrix.
        # does not multiply through by an identity matrix
        return arg

