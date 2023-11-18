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
    def conic_repr(self, expr):
        t = self.add_var()
        stack = concatenate([1.0 / 2.0, t, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, RQuad(dim))
        self.add_linear_term(t)
