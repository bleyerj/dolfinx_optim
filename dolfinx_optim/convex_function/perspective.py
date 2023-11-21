#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   20/11/2023
"""
from dolfinx_optim.utils import concatenate, get_shape, tail, split_affine_expression
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
        variables = self.variables + self._problem_variables

        t = expr[0]

        self.linear_constraints = []
        for cons in self.fun.linear_constraints:
            d = get_shape(cons["expr"])
            if cons["bu"] is not None:
                if get_shape(cons["bu"]) != d:
                    buv = ufl.as_vector([1] * d) * cons["bu"]
                else:
                    buv = cons["bu"]
                self.add_ineq_constraint(cons["expr"] - buv * t, bu=0)
            if cons["bl"] is not None:
                if get_shape(cons["bl"]) != d:
                    blv = ufl.as_vector([1] * d) * cons["bl"]
                else:
                    blv = cons["bl"]
                self.add_ineq_constraint(cons["expr"] - blv * t, bl=0)

        for cons in self.conic_constraints:
            linear, constant = split_affine_expression(
                cons["expr"],
                variables,
            )
            cons["expr"] = sum(linear) + t * constant

        self.add_ineq_constraint(
            sum(self.fun._linear_objective) - t, bu=0.0, name="epigraph-constraint"
        )
