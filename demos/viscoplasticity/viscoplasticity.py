#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steady-state 2D flow of a viscoplastic Bingham fluid in a lid-driven cavity.

See more details in (Bleyer, 2015 and 2016)
https://dx.doi.org/10.1016/j.cma.2017.11.006
https://dx.doi.org/10.1016/j.cma.2014.10.008

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh, io
import ufl
from ufl import dot, grad
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import QuadraticTerm, L2Norm

N = 25
domain = mesh.create_unit_square(
    MPI.COMM_WORLD, N, N, diagonal=mesh.DiagonalType.crossed
)

# fluid viscosity and yield stress
# with unit length and unit imposed viscosity, the Bingham number
# is Bi = tau0/mu
mu, tau0 = fem.Constant(domain, 1.0), fem.Constant(domain, 20.0)


# get top boundary and remaining part of the cavity
def sides(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0)


def top(x):
    return np.isclose(x[1], 1.0) & (x[0] > 0.01) & (x[0] < 0.99)


def middle(x):
    return np.isclose(x[0], 0.5)


prob = MosekProblem(domain, "Viscoplastic fluid")

# P2 interpolation for velocity
V = fem.functionspace(domain, ("P", 2, (2,)))
side_dofs = fem.locate_dofs_geometrical(V, sides)
top_dofs = fem.locate_dofs_geometrical(V, top)
Vx, _ = V.sub(0).collapse()
middle_dofs = fem.locate_dofs_geometrical((V.sub(0), Vx), middle)[1]


bc = [
    fem.dirichletbc(fem.Constant(domain, (1.0, 0.0)), top_dofs, V),
    fem.dirichletbc(fem.Constant(domain, (0.0, 0.0)), side_dofs, V),
]
u = prob.add_var(V, bc=bc)

# P1 interpolation for pressure (Lagrange multiplier)
Vp = fem.functionspace(domain, ("P", 1))


# Mass conservation condition
p = ufl.TestFunction(Vp)
mass_conserv = p * ufl.div(u) * ufl.dx
prob.add_eq_constraint(mass_conserv)


def strain(v):
    """
    Express strain tensor in vectorial notation.

    such that ||strain(v)||_2^2 = inner(E, E) with E = sym(grad(v)).
    """
    E = ufl.sym(grad(v))
    return ufl.as_vector([E[0, 0], E[1, 1], ufl.sqrt(2) * E[0, 1]])


visc = QuadraticTerm(strain(u), 2)
plast = L2Norm(strain(u), 2)

# add viscous term mu*||strain||_2^2 (factor 2 because 1/2 in QuadraticTerm)
prob.add_convex_term(2 * mu * visc)
# add plastic term sqrt(2)*tau0*||strain||_2
prob.add_convex_term(np.sqrt(2) * tau0 * plast)

prob.optimize()


plt.figure()
ux = u.sub(0).collapse()
ux_vals = ux.x.array[middle_dofs]
y_dofs = Vx.tabulate_dof_coordinates()[middle_dofs, 1]
plt.plot(ux_vals, y_dofs, label="present computation")
Bi = int(float(tau0 / mu))
Bi_data_values = [0, 2, 5, 20]
try:
    i = Bi_data_values.index(Bi)
    data = np.loadtxt("viscoplastic_data.csv", skiprows=1)
    plt.plot(
        data[:, i + 1],
        data[:, 0],
        "o",
        markersize=4,
        label="solution from [Bleyer et al., 2015]",
    )
except:
    pass
plt.legend()
plt.xlabel("$y$ coordinate")
plt.ylabel("Velocity $u_x$")
plt.show()

# compare horizontal velociy profile against previous solutions for some Bingham numbers
# along vertical line x=0.5
# plt.figure()
# y = np.linspace(0, 1, 200)
# u_mid = [u(0.5, yi)[0] for yi in y]
# plt.plot(u_mid, y, label="present computation")
# plt.legend()
# plt.xlabel("$y$ coordinate")
# plt.ylabel("Velocity $u_x$")
# plt.show()
