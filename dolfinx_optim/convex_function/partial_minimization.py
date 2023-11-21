#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   21/11/2023
"""
from dolfinx_optim.utils import concatenate, get_shape
from dolfinx_optim.convex_function import ConvexTerm
import ufl


class PartialMinimization(ConvexTerm):
    def __init__(self, fun, indices):
        self.fun = fun
        expr = fun.operand
        self.indices = indices
        d = get_shape(expr)
        new_op = concatenate([expr[i] for i in range(d) if i not in indices])
        super().__init__(new_op, fun.deg_quad)

    def conic_repr(self, expr):
        self.fun._apply_conic_representation()
        self.copy(self.fun)

        x_free = ufl.as_vector([self.fun.operand[i] for i in self.indices])
        y = self.add_var(len(self.indices))
        old_obj = sum(self.fun._linear_objective)
        if old_obj != 0:
            obj = ufl.replace(old_obj, {x_free: y})
            self.add_linear_term(obj)

        self.add_eq_constraint(x_free - y)
