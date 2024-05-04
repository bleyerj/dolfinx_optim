#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   20/11/2023
"""
from dolfinx_optim.utils import (
    concatenate,
    get_shape,
    split_affine_expression,
    reshape,
)
from dolfinx_optim.convex_function import ConvexTerm
import ufl


class Perspective(ConvexTerm):
    def __init__(self, t, fun):
        self.fun = fun
        expr = fun.operand
        stack = concatenate([t, expr])
        super().__init__(stack, fun.deg_quad)

    def conic_repr(self, expr):
        self.fun._apply_conic_representation()
        self.copy(self.fun)

        t = expr[0]
        _, _, t0 = split_affine_expression(
            t,
            self.operand,
            self.variables,
        )

        self.linear_constraints = []
        for cons in self.fun.linear_constraints:
            new_cons = cons
            d = get_shape(cons["expr"])
            if cons["bu"] is not None:
                buv = reshape(cons["bu"], d)
                new_cons["expr"] -= buv * t
                new_cons["bu"] = buv * t0
            if cons["bl"] is not None:
                blv = reshape(cons["bl"], d)
                new_cons["expr"] -= blv * t
                new_cons["bl"] = blv * t0
            self.linear_constraints.append(new_cons)

        for cons in self.conic_constraints:
            linear_op, linear_var, constant = split_affine_expression(
                cons["expr"],
                self.fun.operand,
                self.variables,
            )
            cons["expr"] = cons["expr"] - constant + (t + t0) * constant
        self.add_ineq_constraint(t + t0, bl=0)
