#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some useful utility functions.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import numpy as np
from ufl import shape, as_matrix, as_vector, outer, cross, sqrt, dot, inner, avg
import ufl
from dolfinx import fem


def get_shape(expr):
    expr_shape = shape(expr)
    if len(expr_shape) == 0:  # scalar constraint
        dim = 0
    elif len(expr_shape) == 1:
        dim = expr_shape[0]
    else:
        dim = expr_shape
    return dim


def to_list(a, n=1):
    """Transform `a` to list of length `n`."""
    if not isinstance(a, list):
        return [a] * n
    else:
        return a


def to_vect(X, symmetric=False):
    if symmetric:
        raise NotImplementedError
    else:
        d1, d2 = get_shape(X)
        return as_vector([X[i, j] for i in range(d1) for j in range(d2)])


def to_mat(X, symmetric=False, shape=None):
    if symmetric:
        raise NotImplementedError
    else:
        if shape is None:  # assumes a square matrix
            d1 = int(np.sqrt(len(X)))
            d2 = d1
        else:
            d1, d2 = shape
            assert (
                len(X) == d1 * d2
            ), f"Vector of size {len(X)} cannot be written as {d1}x{d2}."
        return as_matrix([[X[j + d2 * i] for j in range(d2)] for i in range(d1)])


def concatenate(vectors):
    """Concatenate vectors."""
    concat = []
    for v in vectors:
        if shape(v) == ():
            concat += [v]
        else:
            concat += [v[i] for i in range(len(v))]
    return as_vector(concat)


def vstack(arrays):
    """Vertical stack of vectors/matrix."""
    if all([len(a.ufl_shape) <= 1 for a in arrays]):
        return concatenate(arrays)

    shapes = [shape(a)[0] if len(shape(a)) == 1 else shape(a)[1] for a in arrays]
    assert len(set(shapes)) == 1, "Arrays must have matching dimensions."
    final_array = []
    for arr in arrays:
        if len(shape(arr)) == 2:
            final_array += [
                [arr[i, j] for j in range(shape(arr)[1])] for i in range(shape(arr)[0])
            ]
        else:
            final_array += [[arr[i] for i in range(len(arr))]]
    return as_matrix(final_array)


def hstack(arrays):
    """Vertical stack of vectors/matrix."""
    shapes = [a.ufl_shape for a in arrays]
    if all([len(s) <= 1 for s in shapes]):
        return as_matrix([[a[i] for a in arrays] for i in range(shapes[0][0])])

    shapes = [shape(a)[0] for a in arrays]
    assert len(set(shapes)) == 1, "Arrays must have matching dimensions."
    final_array = []
    for arr in arrays:
        if len(shape(arr)) == 2:
            final_array += [
                [arr[j, i] for j in range(shape(arr)[0])] for i in range(shape(arr)[1])
            ]
        else:
            final_array += [[arr[i] for i in range(len(arr))]]
    final_array = list(map(list, zip(*final_array)))
    return as_matrix(final_array)


def block_matrix(M):
    """Horizontal stack of vectors."""
    return vstack([hstack(m) for m in M])


def get_slice(x, start=0, end=None, step=None):
    """Get a slice x[start:end:step]."""
    dim = shape(x)[0]
    if end is None:
        end = dim
    elif end < 0:
        end = dim + end
    elif end == 0:
        end = 1
    if step is None:
        step = 1
    return as_vector([x[i] for i in range(start, end, step)])


def tail(x):
    """Get the tail x[1:] of a vector."""
    return get_slice(x, start=1)


def split_affine_expression(expr, operand, variables):
    if expr == 0:
        return 0, 0, 0
    else:
        e = ufl.variable(operand)
        new_expr = ufl.replace(expr, {operand: e})
        new_expr = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(
            new_expr
        )
        linear_op = ufl.algorithms.apply_derivatives.apply_derivatives(
            ufl.diff(new_expr, e)
        )
        linear_var = [
            ufl.algorithms.apply_derivatives.apply_derivatives(
                ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(
                    ufl.diff(expr, v)
                )
            )
            for v in variables
            if not isinstance(v, fem.Constant)
        ]
        d = get_shape(operand)
        if d == 0:
            z = 0
        else:
            z = ufl.as_vector([0] * d)
        constant = ufl.replace(
            ufl.replace(expr, {operand: z}), {v: 0 * v for v in variables}
        )

        return (
            ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(linear_op),
            linear_var,
            ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(constant),
        )


def reshape(x, d):
    if get_shape(x) != d:
        return ufl.as_vector([1] * d) * x
    else:
        return x
