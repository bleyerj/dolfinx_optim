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
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import (
    SpectralNorm,
    NuclearNorm,
    LambdaMax,
    FrobeniusNorm,
)


def generate_3D_Cheeger_problem():
    N = 1
    domain = mesh.create_unit_cube(
        MPI.COMM_WORLD,
        N,
        N,
        N,
        cell_type=mesh.CellType.hexahedron,
    )

    def border(x):
        return np.full_like(x[0], True)

    V0 = fem.functionspace(domain, ("DG", 0, (3, 3)))
    dofs = fem.locate_dofs_geometrical(V0, border)

    X = np.random.rand(3, 3)
    X = 0.5 * (X + X.T)
    bc = fem.dirichletbc(X, dofs, V0)

    prob = MosekProblem(domain, "Test")
    g = prob.add_var(V0, bc=bc)

    return prob, g, X


norms = [LambdaMax, SpectralNorm, NuclearNorm, FrobeniusNorm]
fun_eval = [
    lambda X: max(np.linalg.eig(X)[0]),
    lambda X: np.linalg.norm(X, ord=2),
    lambda X: np.linalg.norm(X, ord="nuc"),
    lambda X: np.linalg.norm(X, ord="fro"),
]


@pytest.mark.parametrize("norm, value", zip(norms, fun_eval))
def test_norms(norm, value):
    prob, g, X = generate_3D_Cheeger_problem()
    pi = norm(g, 1)
    prob.add_convex_term(pi)
    pobj, dobj = prob.optimize()
    assert np.isclose(pobj, value(X))
    assert np.isclose(dobj, value(X))


# test_norms(LambdaMax, fun_eval[0])
