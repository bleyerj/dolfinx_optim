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


class Product(Cone):
    """Direct product of cones."""

    def __init__(self, list_K):
        self.cones = list_K
        self.dim = [K.dim for K in self.cones]


class SDP(Cone):
    """
    The cone of positive semi-definite matrices.

    :math:`\\mathcal{{S}}=\\{{\\boldsymbol{{X}} \\text{ s.t. }
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


class Pow(Cone):
    """The primal power cone.

    :math:`\\mathcal{P}_{{\\alpha,1-\\alpha}}=\\{{x=(x_0,x_1,\\bar{{x}})
    \\text{ s.t. } \\|\\bar{{x}}\\|_2^2 \\leq x_0^\\alpha x_1^{{1-\\alpha}}\\}}`

    Parameters
    ----------
    dim : int
        [description]
    alpha : float
        Power-cone exponent, must be between 0 and 1.
    """

    def __init__(self, dim: int, alpha: float):
        self.dim = dim
        assert 0 < float(alpha) < 1, "Exponent alpha must be between 0 and 1."
        self.alpha = alpha
        self.type = "ppow"


class Exp(Cone):
    """The primal exponential cone."""

    def __init__(self, dim: int):
        self.dim = dim
        self.type = "pexp"
