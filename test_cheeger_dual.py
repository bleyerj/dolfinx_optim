#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test case on a square Cheeger problem.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, io
import ufl
from ufl import dot, grad
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import L2Norm, L2Ball


N = 50
domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    N,
    N,
    diagonal=mesh.DiagonalType.crossed,
    cell_type=mesh.CellType.quadrilateral,
)


def border(x):  # noqa
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[0], 0))


# boundaries
boundaries = [(1, border)]  # on top     = 6

# create facet tags
facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for marker, locator in boundaries:
    facets = mesh.locate_entities_boundary(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(
    domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
)

f = fem.Constant(domain, 1.0)


deg_quad = 2
V = fem.FunctionSpace(domain, ("RT", 1))
dofs = fem.locate_dofs_topological(V, fdim, facet_tag.find(1))
sig0 = fem.Function(V)
bc = fem.dirichletbc(sig0, dofs)

prob = MosekProblem(domain, "Test")
lamb, sig = prob.add_var([1, V], bc=[None, bc])

dx = ufl.Measure("dx", domain=domain)

V0 = fem.FunctionSpace(domain, ("DG", 0))
u_ = ufl.TestFunction(V0)
equilibrium = (lamb * f - ufl.div(sig)) * u_ * dx

prob.add_eq_constraint(A=equilibrium, name="u")

prob.add_obj_func(lamb * dx)

crit = L2Ball(sig, deg_quad)
prob.add_convex_term(crit)

prob.optimize(sense="max")

u = prob.get_lagrange_multiplier("u")
prob.get_solution_info()

with io.XDMFFile(MPI.COMM_WORLD, "limit_analysis/sig.xdmf", "w") as file:
    file.write_mesh(sig.function_space.mesh)
    file.write_function(sig)
    file.write_function(u)
