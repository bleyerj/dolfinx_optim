#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convex optimization formulation of the Cheeger problem.

The Cheeger problem consists in finding the subset of a domain
:math:`Omega` which minimizes the ratio of perimeter over area.

Geometry can be either a unit square or a star-shaped polynom.
Various discretization are available (DG0, DG1, CG1, CG2,...).
The default norm for the classical Cheeger problem is the L2-norm of
the gradient, other anisotropic norms like L1 or Linf are available.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import numpy as np
from mpi4py import MPI
import pyvista
import ufl
from ufl import dot, grad
from dolfinx import fem, mesh, io, plot
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import L1Norm, L2Norm, LinfNorm, LpNorm


def LpNorm4(*args):
    return LpNorm(*args, 4)


norms = {"l2": L2Norm, "l1": L1Norm, "linf": LinfNorm, "lp4": LpNorm4}


def Cheeger_pb(N: int, interp: str, degree: int, norm_type: str = "l2"):
    """Cheeger problem.

    Parameters
    ----------
    N : int
        mesh resolution
    interp : str
        interpolation type
    degree : int
        interpolation degree
    norm_type : {"l2", "l1", "linf"}
        norm used in the Cheeger problem objective
    """
    # Define geometry
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        N,
        N,
        cell_type=mesh.CellType.quadrilateral,
    )

    def border(x):
        return (
            np.isclose(x[1], 0)
            | np.isclose(x[0], 0)
            | np.isclose(x[1], 1)
            | np.isclose(x[0], 1)
        )

    f = fem.Constant(domain, 1.0)

    V = fem.functionspace(domain, (interp, degree))
    dofs = fem.locate_dofs_geometrical(V, border)
    bc = fem.dirichletbc(0.0, dofs, V)

    # Define variational problem
    prob = MosekProblem(domain, "Cheeger problem")
    u = prob.add_var(V, bc=bc)

    # Choose norm type using predefined functions

    # Choose quadrature scheme depending on chosen interpolation degree
    F = norms[norm_type](grad(u), 2 * degree)

    prob.add_convex_term(F)

    # Adds normalization constraint (f, u)=1
    dx = ufl.Measure("dx", domain=domain)

    Pext = dot(f, u) * dx

    prob.add_eq_constraint(A=Pext, b=1.0)

    pobj, dobj = prob.optimize()

    pyvista.set_jupyter_backend("static")

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["Displacement"] = u.x.array
    u_grid.set_active_scalars("Displacement")

    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(u_grid, show_scalar_bar=True, scalars="Displacement", cmap="Blues")
    edges = u_grid.extract_all_edges()
    plotter.add_mesh(edges, color="k", line_width=1)

    plotter.view_xy()
    plotter.screenshot("Cheeger_set.png")
    return pobj


Cheeger_pb(50, "Q", degree=1, norm_type="lp4")
