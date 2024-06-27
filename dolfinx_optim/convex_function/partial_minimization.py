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
    """Performs a partial minimization of function :math:`f(x)` over some coordinates given by a set of `indices`:

    .. math::

        \\underline{f}(x) = \\inf_{z_i \\text{ if } i\in\\texttt{indices}} f(z)

    where :math:`x=(z_j)_{j\\notin \\texttt{indices}}`.
    """

    def __init__(self, fun, indices):
        self.fun = fun
        expr = fun.operand
        self.indices = indices
        d = get_shape(expr)
        new_op = concatenate([expr[i] for i in range(d) if i not in indices])
        super().__init__(new_op, fun.deg_quad)

    def conic_repr(self, expr):
        d = get_shape(self.fun.operand)
        x_free = ufl.as_vector([self.fun.operand[i] for i in self.indices])
        y = self.add_var(len(self.indices))
        mapping = {i: k for k, i in enumerate(self.indices)}
        new_op = ufl.as_vector(
            [
                self.fun.operand[i] if i not in self.indices else y[mapping[i]]
                for i in range(d)
            ]
        )

        self.fun._apply_conic_representation()
        self.fun.change_operand(new_op)

        self.copy(self.fun)

        self.add_eq_constraint(x_free - y)
