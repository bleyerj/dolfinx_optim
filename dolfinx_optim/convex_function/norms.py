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


class L2Norm(ConvexTerm):
    def conic_repr(self, expr):
        print(expr)
        suff = np.random.randint(0, 100)
        t = self.add_var(name=f"t{suff}")
        print("Conic_repr", t.name)
        stack = concatenate([t, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, Quad(dim))
        self.add_linear_term(t)


class L2Ball(ConvexTerm):
    def conic_repr(self, expr):
        stack = concatenate([1, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, Quad(dim))
