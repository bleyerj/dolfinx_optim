#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   20/11/2023
"""
import ufl
from dolfinx_optim.utils import get_shape
from dolfinx_optim.convex_function import ConvexTerm


class InfConvolution(ConvexTerm):
    """Transforms two function :math:`f_1(x)` and :math:`f_2(x)` into their inf-convolution:

    .. math::

        (f_1\square f_2)(x) = \\inf_{x \\text{ s.t. } x=y_1+y_2} f_1(y_1)+f_2(y_2)

    If `indices` are provided, convolution is taken only over this set of indices.
    """

    def __init__(self, fun1, fun2, indices=None):
        self.fun1 = fun1
        self.fun2 = fun2
        self.indices = indices
        d1 = get_shape(fun1.operand)
        d2 = get_shape(fun2.operand)
        if indices is None:
            assert d1 == d2, "Both functions should have the same argument shape."
        else:
            assert d1 >= len(
                indices
            ), "The first function argument must have at least the same length as the indices list."
            assert d2 >= len(
                indices
            ), "The second function argument must have at least the same length as the indices list."
        assert (
            fun1.deg_quad == fun2.deg_quad
        ), "Both functions should have the same quadrature degree."

        expr = fun1.operand
        super().__init__(expr, fun1.deg_quad)

    def conic_repr(self, expr):
        d1 = get_shape(self.fun1.operand)
        d2 = get_shape(self.fun2.operand)
        y1 = self.add_var(d1, name="y1")
        y2 = self.add_var(d2, name="y2")
        if self.indices is None:
            self.add_eq_constraint(expr - y1 - y2, name="inf-conv")
        else:
            self.add_eq_constraint(
                ufl.as_vector([expr[i] - y1[i] - y2[i] for i in self.indices]),
                name="inf-conv",
            )

        self.fun1._apply_conic_representation()
        self.fun2._apply_conic_representation()
        self.fun1.change_operand(y1)
        self.fun2.change_operand(y2)
        for fun in [self.fun1, self.fun2]:
            self.linear_constraints += fun.linear_constraints
            self.conic_constraints += fun.conic_constraints
            self.variables += fun.variables
            self.variable_names += fun.variable_names
            self.ux += fun.ux
            self.lx += fun.lx
            self.cones += fun.cones
            self.scale_factor = 1.0
            self._linear_objective += [
                fun.scale_factor * obj for obj in fun._linear_objective
            ]
