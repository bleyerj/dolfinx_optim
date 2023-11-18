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
import numpy as np
import ufl


class AbsValue(ConvexTerm):
    """Define the absolute value function :math:`|x|`."""

    def conic_repr(self, expr):
        assert get_shape(expr) == 0, "Absolute value applies only to scalar function"
        t = self.add_var()
        self.add_ineq_constraint(expr - t, bu=0.0)
        self.add_ineq_constraint(-expr - t, bu=0.0)
        self.add_linear_term(t)


class L2Norm(ConvexTerm):
    def conic_repr(self, expr):
        t = self.add_var()
        stack = concatenate([t, expr])
        print(stack)
        dim = get_shape(stack)
        print("Dim", dim)
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


class L2Ball(ConvexTerm):
    def conic_repr(self, expr):
        stack = concatenate([1, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, Quad(dim))
