#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Definition of various norms and balls.

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   17/11/2023
"""
from dolfinx_optim.utils import concatenate, get_shape
from dolfinx_optim.cones import Quad, RQuad, Pow
from dolfinx_optim.convex_function import ConvexTerm
from dolfinx_optim.convex_function.epigraph import Epigraph
import numpy as np
import ufl


def L2Ball(*args, k=1.0):
    return Epigraph(k, L2Norm(*args))


def L1Ball(*args, k=1.0):
    return Epigraph(k, L1Norm(*args))


def LinfBall(*args, k=1.0):
    return Epigraph(k, LinfNorm(*args))


def LpBall(*args, k=1.0):
    return Epigraph(k, LpNorm(*args))


class L2Norm(ConvexTerm):
    def conic_repr(self, expr):
        t = self.add_var(name="L2Norm_t")
        stack = concatenate([t, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, Quad(dim))
        self.add_linear_term(t)


class L1Norm(ConvexTerm):
    """Define the L1-norm function :math:`||x||_1`."""

    def conic_repr(self, expr):
        d = get_shape(expr)
        z = self.add_var(d)
        self.add_ineq_constraint(expr - z, bu=0.0)
        self.add_ineq_constraint(-expr - z, bu=0.0)
        if d == 0:
            obj = z
        else:
            obj = sum(z)
        self.add_linear_term(obj)


class LinfNorm(ConvexTerm):
    """Define the Linf-norm function :math:`||x||_\infty`."""

    def conic_repr(self, expr):
        d = get_shape(expr)
        z = self.add_var()
        if d == 0:
            e = 1.0
        else:
            e = ufl.as_vector([1] * d)
        self.add_ineq_constraint(expr - z * e, bu=0.0)
        self.add_ineq_constraint(-expr - z * e, bu=0.0)
        self.add_linear_term(z)


class LpNorm(ConvexTerm):
    """Define the Linf-norm function :math:`||x||_p`."""

    def __init__(self, operand, deg_quad, p):
        super().__init__(operand, deg_quad, parameters=(p,))

    def conic_repr(self, expr, p):
        t = self.add_var()
        d = get_shape(expr)
        if d == 0:
            stack = concatenate([t, t, expr])
            self.add_conic_constraint(stack, Pow(3, 1 / p))
        else:
            r = self.add_var(d)
            self.add_eq_constraint(sum(r) - t)
            for i in range(d):
                stack = concatenate([r[i], t, expr[i]])
                self.add_conic_constraint(stack, Pow(3, 1 / p))
        self.add_linear_term(t)
