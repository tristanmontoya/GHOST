import numpy as np
from scipy import special


class PolynomialSpace:
    """
    The local polynomial space

    Attributes
    ----------

    d : int
        spatial dimension

    p : int
        degree of polynomial space

    Np : int
        dimension of polynomial space

    basis : str
        choice of basis - currently only 'legendre-normalized'

    type : str
        'total-degree' or 'tensor-product'

    Methods
    -------

    __init__

    vandermonde

    grad_vandermonde

    """

    def __init__(self, d, p, basis, type):
        self.d = d
        self.p = p
        if type == 'tensor-product':
            self.Np = self.p**self.d
        else:
            self.Np = special.comb(self.p + self.d, self.d, exact=True)
        self.basis = basis

    def vandermonde(self, x):
        if self.d == 1:
            if self.basis == 'legendre-normalized':
                return self._orthonormal_vandermonde_1d(x)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def grad_vandermonde(self, x):
        if self.d == 1:
            if self.basis == 'legendre-normalized':
                return self._orthonormal_grad_vandermonde_1d(x)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def _orthonormal_vandermonde_1d(self, x):
        V = np.polynomial.legendre.legvander(x[:, 0], self.p)

        for j in range(0, self.Np):
            normalization_factor = np.sqrt(2. / (2 * j + 1))
            V[:, j] /= normalization_factor

        return V

    def _orthonormal_grad_vandermonde_1d(self, x):
        Vx = np.zeros([len(x), self.Np])

        for j in range(0, self.Np):
            normalization_factor = np.sqrt(2. / (2 * j + 1))
            dPdx = np.polyder(special.legendre(j))
            Vx[:, j] = dPdx(x[:, 0]) / normalization_factor

        # this is for the 3D array convention -- to be general in terms of
        # dimension, we have an array of matrices giving the vandermonde
        # derivative matrix in each direction

        return np.array([Vx])
