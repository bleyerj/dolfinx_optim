#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   22/11/2023
"""
import ufl
from dolfinx_optim.convex_function import ConvexTerm
from dolfinx_optim.utils import to_mat, get_shape, block_matrix, to_vect, concatenate
from dolfinx_optim.cones import SDP, Quad


class LambdaMax(ConvexTerm):
    """Define the maximum eigenvalue function
    :math:`\\boldsymbol{X}\\in \\mathcal{S}_n \\to
    \\lambda_{max}(\\boldsymbol{X})`."""

    def conic_repr(self, expr):
        t = self.add_var()
        Expr = to_mat(expr, False)
        d, d2 = get_shape(Expr)
        assert d == d2
        Id = ufl.Identity(d)
        self.add_conic_constraint(t * Id - Expr, SDP(d))
        self.add_linear_term(t)


class SpectralRadius(ConvexTerm):
    """Define the spectral radius function
    :math:`\\boldsymbol{X}\\in \\mathcal{S}_n \\to
    \max_i\\{{|\\lambda_i(\\boldsymbol{X})|\\}}`."""

    def conic_repr(self, expr):
        t = self.add_var()
        Expr = to_mat(expr, False)
        d, d2 = get_shape(Expr)
        assert d == d2
        Id = ufl.Identity(d)
        self.add_conic_constraint(t * Id - Expr, SDP(d))
        self.add_conic_constraint(Expr + t * Id, SDP(d))
        self.add_linear_term(t)


class SpectralNorm(ConvexTerm):
    """Define the spectral norm function
    :math:`\\boldsymbol{X}\\in \\mathcal{S}_n \\to
    \sqrt{\\lambda_{max}(\\boldsymbol{X}^T\\boldsymbol{X})}`."""

    def conic_repr(self, expr):
        t = self.add_var()
        Expr = to_mat(expr, False)
        d1, d2 = get_shape(Expr)
        assert d2 <= d1, "Works only for rectangular matrix m x n with n <= m."
        Id1 = ufl.Identity(d1)
        Id2 = ufl.Identity(d1)
        Z = block_matrix([[t * Id1, Expr], [Expr.T, t * Id2]])
        self.add_conic_constraint(Z, SDP(d1 + d2))
        self.add_linear_term(t)


class NuclearNorm(ConvexTerm):
    """Define the nuclear norm function
    :math:`\\boldsymbol{X}\\in \\mathcal{S}_n \\to
    \\text{tr}(\sqrt{\\boldsymbol{X}^T\\boldsymbol{X}})`."""

    def conic_repr(self, expr):
        Expr = to_mat(expr, False)
        d1, d2 = get_shape(Expr)
        U = self.add_var((d1, d1))
        V = self.add_var((d2, d2))

        Z = block_matrix([[U, Expr], [Expr.T, V]])
        self.add_conic_constraint(Z, SDP(d1 + d2))
        self.add_linear_term(ufl.tr(U + V) / 2)


class FrobeniusNorm(ConvexTerm):
    """Define the Froebenius norm function
    :math:`\\boldsymbol{X}\\in \\mathcal{S}_n \\to
    \sqrt{\\text{tr}(\\boldsymbol{X}^T\\boldsymbol{X})}`."""

    def conic_repr(self, expr):
        t = self.add_var()
        d = get_shape(expr)
        stack = concatenate([t, expr])

        self.add_conic_constraint(stack, Quad(d + 1))
        self.add_linear_term(t)
