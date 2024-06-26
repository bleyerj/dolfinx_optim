#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   21/11/2023
"""
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from dolfinx_optim.utils import (
    concatenate,
    to_mat,
    split_affine_expression,
    reshape,
    hstack,
    vstack,
)
from dolfinx_optim.convex_function import ConvexTerm
import ufl


def transpose(A):
    if len(ufl.shape(A)) == 2:
        return A.T
    else:
        return A


class Conjugate(ConvexTerm):
    """Transforms a function :math:`f(x)` into its Legendre-Fenchel conjugate:

    .. math::

        f^*(z) = \\sup_{x} z^Tx - f(x)
    """

    def __init__(self, z, fun):
        self.fun = fun
        super().__init__(z, fun.deg_quad)

    def conic_repr(self, expr):
        self.fun._apply_conic_representation()
        self.fun.change_operand(self.operand)

        c = sum(self.fun._linear_objective)
        constraint = concatenate([-expr] + [0 * v for v in self.fun.variables])
        if c != 0:
            c_op, c_var, _ = split_affine_expression(
                c, self.fun.operand, self.fun.variables
            )
            constraint += concatenate([c_op] + c_var)

        for cons in self.fun.linear_constraints:
            A_op, A_var, b0 = split_affine_expression(
                cons["expr"], self.fun.operand, self.fun.variables
            )
            A = hstack([A_op] + [a for a in A_var])

            d = cons["dim"]

            if cons["bu"] == cons["bl"]:
                b = reshape(cons["bu"], d) - b0
                s = self.add_var(d)
                self.add_linear_term(ufl.dot(s, b))
                constraint += apply_algebra_lowering(transpose(A) * s)
            else:
                if cons["bu"] is not None:
                    bu = reshape(cons["bu"], d) - b0
                    su = self.add_var(d, lx=0)
                    self.add_linear_term(ufl.dot(su, bu))
                    print(transpose(A).ufl_shape, su.ufl_shape)
                    constraint += apply_algebra_lowering(transpose(A) * su)
                if cons["bl"] is not None:
                    bl = reshape(cons["bl"], d) - b0
                    sl = self.add_var(d, lx=0)
                    self.add_linear_term(-ufl.dot(sl, bl))
                    constraint -= apply_algebra_lowering(transpose(A) * sl)

        for cons in self.fun.conic_constraints:
            F_op, F_var, g = split_affine_expression(
                cons["expr"], self.fun.operand, self.fun.variables
            )
            F = hstack([F_op] + [f for f in F_var])
            d = cons["dim"]
            sc = self.add_var(d)
            self.add_linear_term(ufl.dot(sc, g))
            constraint -= apply_algebra_lowering(ufl.dot(F.T, sc))
            if cons["cone"].type == "sdp":
                self.add_conic_constraint(
                    apply_algebra_lowering(to_mat(sc, False)), cons["cone"].dual
                )
            else:
                self.add_conic_constraint(sc, cons["cone"].dual)

        self.add_eq_constraint(constraint)
