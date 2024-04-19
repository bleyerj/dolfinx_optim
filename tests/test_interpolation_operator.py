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
from ufl import dot, grad, inner
from dolfinx_optim.mosek_io import MosekProblem

N = 5
domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    N,
    N,
    cell_type=mesh.CellType.quadrilateral,
)

deg=1
V = fem.functionspace(domain, ("CG", deg, ()))
u = fem.Function(V)
u.interpolate(lambda x: x[0]+2*x[1])

prob = MosekProblem(domain, name="Optimal vault")
u = prob.add_var(V, name="Displacement", lx=u, ux=u)


Vv = fem.functionspace(domain, ("DG", deg - 1, ((2,))))
g = prob.add_var(Vv, name="g")


dx = ufl.Measure("dx")
sig = ufl.TestFunction(Vv)
constraint = inner(sig, (grad(u) - g)) * dx

prob.add_eq_constraint(A=constraint)
prob.add_obj_func(u*dx)

pobj, dobj = prob.optimize()

assert np.allclose(g.x.array, np.array([1, 2]*(len(g.x.array)//2)))
assert np.isclose(pobj, 1.5)