#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cones supported by Mosek.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""


class Cone:
    """A generic class for cones."""


class Quad(Cone):
    """
    The quadratic cone.

    :math:`\\mathcal{Q}=\\{{x=(x_0,\\bar{{x}})
    \\text{ s.t. } \\|\\bar{{x}}\\|_2 \\leq x_0\\}}`

    Parameters
    ----------
    dim : int, optional
        dimension of the cone, by default 1
    """

    def __init__(self, dim: int = 1):
        self.dim = dim
        self.type = "quad"

    @property
    def dual(self):
        return self


class RQuad(Cone):
    """
    The rotated quadratic cone.

    :math:`\\mathcal{Q}_r=\\{{x=(x_0,x_1,\\bar{{x}})
    \\text{ s.t. } \\|\\bar{{x}}\\|_2^2 \\leq 2x_0x_1\\}}`

    Parameters
    ----------
    dim : int, optional
        dimension of the cone, by default 1
    """

    def __init__(self, dim: int = 1):
        self.dim = dim
        self.type = "rquad"

    @property
    def dual(self):
        return self


class Product(Cone):
    """Direct product of cones."""

    def __init__(self, list_K):
        self.cones = list_K
        self.dim = [K.dim for K in self.cones]


class SDP(Cone):
    """
    The cone of positive semi-definite matrices.

    :math:`\\mathcal{{S}}=\\{{\\boldsymbol{{X}}\\in \\mathbb{R}^{n\times n} \\text{ s.t. }
    \\boldsymbol{{X}}=\\boldsymbol{{X}}^T \\text{ and }
    \\boldsymbol{{X}}\\succeq 0\\}}`


    Parameters
    ----------
    dim : int
        Dimension :math:`n` of the PSD :math:`n\\times n` matrix
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.type = "sdp"

    @property
    def dual(self):
        return self


class Pow(Cone):
    """The primal power cone.

    :math:`\\mathcal{P}_{{\\alpha,1-\\alpha}}=\\{{x=(x_0,x_1,\\bar{{x}})
    \\text{ s.t. } \\|\\bar{{x}}\\|_2^2 \\leq x_0^\\alpha x_1^{{1-\\alpha}}\\}}`

    Parameters
    ----------
    dim : int
        dimension of the cone (>= 3)
    alpha : float
        Power-cone exponent, must be between 0 and 1.
    """

    def __init__(self, dim: int, alpha: float):
        self.dim = dim
        assert dim >= 3, "Dimension of the cone should be at least 3."
        assert 0 < float(alpha) < 1, "Exponent alpha must be between 0 and 1."
        self.alpha = alpha
        self.type = "ppow"

    @property
    def dual(self):
        """The dual power cone."""

        p = Pow(self.dim, self.alpha)
        p.type = "dpow"
        return p


class Exp(Cone):
    """The primal exponential cone.

    :math:`\\mathcal{K}_\\text{exp} = \\{{x=(x_0,x_1,x_2) \\text{ s.t. }
    x_0 \\geq x_1\\exp(x_2/x_1), \\:\\: x_0,x_1\\geq 0\\}}`
    """

    def __init__(self):
        self.dim = 3
        self.type = "pexp"

    @property
    def dual(self):
        """The dual exponential cone."""
        p = Exp()
        p.type = "dexp"
        return p
