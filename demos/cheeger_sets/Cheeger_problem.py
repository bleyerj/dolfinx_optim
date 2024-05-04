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
from dolfinx_optim.convex_function import L1Norm, L2Norm, LinfNorm

norms = {"l2": L2Norm, "l1": L1Norm, "linf": LinfNorm}


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
        return np.isclose(x[1], 0) | np.isclose(x[0], 0)

    f = fem.Constant(domain, 1.0)

    V = fem.functionspace(domain, (interp, degree))
    dofs = fem.locate_dofs_geometrical(V, border)
    bc = fem.dirichletbc(0.0, dofs, V)

    # Define variational problem
    prob = MosekProblem(domain, "Cheeger problem")
    u = prob.add_var(V, bc=bc)

    # Choose norm type using predefined functions

    # Choose quadrature scheme depending on chosen interpolation degree
    F = norms[norm_type](grad(u), degree)

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
    warped = u_grid.warp_by_scalar("Displacement", factor=5)

    plotter = pyvista.Plotter()
    plotter.add_mesh(
        warped,
        show_scalar_bar=True,
        scalars="Displacement",
    )
    edges = warped.extract_all_edges()
    plotter.add_mesh(edges, color="k", line_width=1)
    pyvista.start_xvfb(wait=0.1)
    plotter.show()
    plotter.screenshot("Cheeger_set.png")
    return pobj


Cheeger_pb(50, "CG", degree=2, norm_type="l2")
