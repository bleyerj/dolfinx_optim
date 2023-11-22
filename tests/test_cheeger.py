#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test case on a square Cheeger problem.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl
from ufl import dot, grad
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import (
    L2Norm,
    L1Norm,
    L1Ball,
    L2Ball,
    LinfNorm,
    AbsValue,
    LpNorm,
    Epigraph,
    Perspective,
    InfConvolution,
    Conjugate,
)


def generate_Cheeger_problem():
    N = 1
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        N,
        N,
        cell_type=mesh.CellType.quadrilateral,
    )

    def border(x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[0], 0))

    f = fem.Constant(domain, 1.0)

    V = fem.FunctionSpace(domain, ("CG", 1))
    dofs = fem.locate_dofs_geometrical(V, border)
    bc = fem.dirichletbc(0.0, dofs, V)

    prob = MosekProblem(domain, "Test")
    # u = prob.add_var(V, bc=bc)

    V0 = fem.FunctionSpace(domain, ("DG", 0))
    u, t = prob.add_var([V, V0], bc=[bc, None], name=["u", "t"], lx=[None, 0])

    dx = ufl.Measure("dx", domain=domain)

    Pext = dot(f, u) * dx

    prob.add_eq_constraint(A=Pext, b=1.0)
    return prob, u, t, dx


def L4Norm(*args):
    return LpNorm(*args, 4.0)


norms = [L2Norm, LinfNorm, L1Norm, L4Norm]

x0 = np.full(2, 2.0)
fun_eval = [
    np.linalg.norm(x0, 2),
    np.linalg.norm(x0, np.inf),
    np.linalg.norm(x0, 1),
    np.linalg.norm(x0, 4.0),
]


@pytest.mark.parametrize("norm, value", zip(norms, fun_eval))
def test_norms(norm, value):
    prob, u, t, dx = generate_Cheeger_problem()
    pi = norm(grad(u), 1)
    prob.add_convex_term(pi)
    pobj, dobj = prob.optimize()
    assert np.isclose(pobj, value)
    assert np.isclose(dobj, value)


@pytest.mark.parametrize("norm, value", zip(norms, fun_eval))
def test_epigraphs(norm, value):
    prob, u, t, dx = generate_Cheeger_problem()
    pi = norm(grad(u), 1)
    epi = Epigraph(t, pi)
    prob.add_convex_term(epi)
    with pytest.raises(ValueError):
        pobj, dobj = prob.optimize()
    prob.add_obj_func(t * dx)
    pobj, dobj = prob.optimize()
    assert np.isclose(pobj, value)
    assert np.isclose(dobj, value)


@pytest.mark.parametrize("norm, value", zip(norms, fun_eval))
def test_perspectives(norm, value):
    prob, u, t, dx = generate_Cheeger_problem()
    pi = norm(grad(u), 1)
    persp = Perspective(t, pi)
    prob.add_convex_term(persp)
    pobj, dobj = prob.optimize()
    assert np.isclose(pobj, value)
    assert np.isclose(dobj, value)


@pytest.mark.parametrize("norm, value", zip(norms, fun_eval))
def test_infconvolution(norm, value):
    prob, u, t, dx = generate_Cheeger_problem()
    pi = norm(grad(u), 1)
    pi2 = 10 * norm(grad(u), 1)
    infc = InfConvolution(pi, pi2)
    prob.add_convex_term(infc)
    pobj, dobj = prob.optimize()
    assert np.isclose(pobj, value)
    assert np.isclose(dobj, value)
