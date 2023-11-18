#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Definition of various norms and balls.

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   17/11/2023
"""
from dolfinx_optim.utils import concatenate, get_shape
from dolfinx_optim.cones import Quad, RQuad
from dolfinx_optim.convex_function import ConvexTerm


class QuadraticTerm(ConvexTerm):
    """Define the quadratic function :math:`\\frac{1}{2}x^T x`.

    Parameters
    ----------
    x : UFL expression
        optimization variable
    """

    def conic_repr(self, expr):
        t = self.add_var()
        stack = concatenate([1.0 / 2.0, t, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, RQuad(dim))
        self.add_linear_term(t)


class QuadOverLin(ConvexTerm):
    """Define the quadratic over linear function :math:`(t,x) \to x^T x/t`.

    Parameters
    ----------
    expr : UFL expression
        expr = (t, x)
    """

    def conic_repr(self, expr):
        y = self.add_var()
        stack = concatenate([y, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, RQuad(dim))
        self.add_linear_term(2 * y)
