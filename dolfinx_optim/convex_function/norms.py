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
from dolfinx_optim.convex_function.partial_minimization import PartialMinimization
import numpy as np
import ufl


def to_ball(norm, k, *args):
    return PartialMinimization(Epigraph(k, norm(*args)), [0])
    # return Epigraph(k, norm(*args))


# def L2Ball(*args, k=1.0):
#     return to_ball(L2Norm, k, *args)


# def L1Ball(*args, k=1.0):
#     return to_ball(L1Norm, k, *args)


# def LinfBall(*args, k=1.0):
#     return to_ball(LinfNorm, k, *args)


def LpBall(*args, k=1.0):
    return to_ball(LpNorm, k, *args)


class L1Ball(ConvexTerm):
    """Define the L1-norm ball constraint :math:`||x||_1 \leq 1`."""

    def conic_repr(self, expr):
        d = get_shape(expr)
        z = self.add_var(d)
        self.add_ineq_constraint(expr - z, bu=0.0)
        self.add_ineq_constraint(-expr - z, bu=0.0)
        if d == 0:
            obj = z
        else:
            obj = sum(z)
        self.add_ineq_constraint(obj, bu=1.0)


class L2Ball(ConvexTerm):
    """Define the L2-norm ball constraint :math:`||x||_2 \leq 1`."""

    def conic_repr(self, expr):
        stack = concatenate([1.0, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, Quad(dim))


class LinfBall(ConvexTerm):
    """Define the Linf-norm ball constraint :math:`||x||_\infty \leq 1`."""

    def conic_repr(self, expr):
        self.add_ineq_constraint(expr, bu=1.0)
        self.add_ineq_constraint(-expr, bu=1.0)


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


class L2Norm(ConvexTerm):
    """Define the L2-norm function :math:`||x||_2`."""

    def conic_repr(self, expr):
        t = self.add_var()
        stack = concatenate([t, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, Quad(dim))
        self.add_linear_term(t)


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
    """Define the Lp-norm function :math:`||x||_p`."""

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


# class LpBall(ConvexTerm):
#     """Define the Lp-ball constraint :math:`||x||_p <= 1.0`."""

#     def __init__(self, operand, deg_quad, p):
#         super().__init__(operand, deg_quad, parameters=(p,))

#     def conic_repr(self, expr, p):
#         d = get_shape(expr)
#         if d == 0:
#             stack = concatenate([1.0, 1.0, expr])
#             self.add_conic_constraint(stack, Pow(3, 1 / p))
#         else:
#             r = self.add_var(d)
#             self.add_eq_constraint(sum(r), b=1.0)
#             for i in range(d):
#                 stack = concatenate([r[i], 1.0, expr[i]])
#                 self.add_conic_constraint(stack, Pow(3, 1 / p))
