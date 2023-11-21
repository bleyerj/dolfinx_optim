#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   20/11/2023
"""
from dolfinx_optim.utils import concatenate, split_affine_expression
from dolfinx_optim.convex_function import ConvexTerm


class Epigraph(ConvexTerm):
    def __init__(self, t, fun):
        self.fun = fun
        expr = fun.operand
        stack = concatenate([t, expr])
        super().__init__(stack, fun.deg_quad)

    def conic_repr(self, expr):
        self.fun._apply_conic_representation()
        self.copy(self.fun)
        self._linear_objective = []

        t = expr[0]
        _, _, t0 = split_affine_expression(t, self.operand, self.variables)
        self.add_ineq_constraint(
            sum(self.fun._linear_objective) - t, bu=t0, name="epigraph-constraint"
        )
