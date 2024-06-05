---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Viscoplastic fluid

We consider the steady-state 2D flow of a viscoplastic Bingham fluid in a lid-driven cavity.

See more details in (Bleyer, 2015 and 2016)
- https://dx.doi.org/10.1016/j.cma.2017.11.006
- https://dx.doi.org/10.1016/j.cma.2014.10.008
$\newcommand{\bu}{\boldsymbol{u}}\newcommand{\be}{\boldsymbol{e}}\newcommand{\dOm}{\text{d}\,\Omega}\newcommand{\bD}{\boldsymbol{D}}$

+++

We first import the relevant modules and create a square mesh.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import QuadraticTerm, L2Norm

path = Path(".")

N = 75
domain = mesh.create_unit_square(
    MPI.COMM_WORLD, N, N, diagonal=mesh.DiagonalType.crossed
)
```

We define no-slip boundary conditions on the bottom and lateral sides and impose a unit horizontal velocity $\bu=\be_x$ on the top boundary.

```{code-cell} ipython3
# get top boundary and remaining part of the cavity
def sides(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0)


def top(x):
    return np.isclose(x[1], 1.0) & (x[0] > 0.01) & (x[0] < 0.99)


def middle(x):
    return np.isclose(x[0], 0.5)



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
```

We now start defining the optimization problem by adding the velocity $\bu$ as an optimization variable on a $\mathbb{P}_2$ function space, constrained to the previously defined boundary conditions. We also enforce the mass conservation condition by adding the following constraint: 

$$
\int_\Omega p\:\text{div}(\bu)\: \dOm = 0 \quad \forall p\in V_p
$$

where $V_p$ is the pressure space for pressure test functions $p$ (used here as Lagrange multipliers). We use a Lagrange $\mathbb{P}_1$ function space for its discretization, thereby corresponding to a Taylor-Hood mixed formulation. 

```{code-cell} ipython3
prob = MosekProblem(domain, "Viscoplastic fluid")

u = prob.add_var(V, bc=bc)

# mass conservation condition
# P1 interpolation for pressure (Lagrange multiplier)
Vp = fem.functionspace(domain, ("P", 1))
p = ufl.TestFunction(Vp)
mass_conserv = p * ufl.div(u) * ufl.dx
prob.add_eq_constraint(mass_conserv)
```

We now define the fluid viscosity $\mu$ and yield stress $\tau_0$. The Bingham number $\text{Bi}=\tau_0 L/\mu V$ represents the proportion of plastic behavior with respect to the viscous behavior. We also introduce a vector representation for the strain rate field such that:

$$
\|\boldsymbol{d}|_2^2 = \sum d_i^2 = \bD:\bD=D_{ij}D_{ij}
$$
where $\bD=\text{sym}(\nabla \bu)$ the strain rate tensor and $\boldsymbol{d}$ its vectorial representation. We then define the viscous potential $\mu\bD:\bD$ with a `QuadraticTerm` and the plastic potential $\sqrt{2}\tau_0\sqrt{\bD:\bD}$ with a `L2Norm`.

```{code-cell} ipython3
mu, tau0 = fem.Constant(domain, 1.0), fem.Constant(domain, 20.0)

def strain(v):
    D = ufl.sym(ufl.grad(v))
    return ufl.as_vector([D[0, 0], D[1, 1], ufl.sqrt(2) * D[0, 1]])


visc = QuadraticTerm(strain(u), 2)
plast = L2Norm(strain(u), 2)

# add viscous term mu*||strain||_2^2 (factor 2 because 1/2 in QuadraticTerm)
prob.add_convex_term(2 * mu * visc)
# add plastic term sqrt(2)*tau0*||strain||_2
prob.add_convex_term(np.sqrt(2) * tau0 * plast)
```

The problem is then solved. Note that we slightly relax the default solver tolerance.

```{code-cell} ipython3
prob.parameters["log_level"] = 0 # deactivate solver output
prob.parameters["tol_rel_gap"] = 1e-6
prob.optimize()
```

We finally compare the horizontal velocity profile against previous solutions for some Bingham numbers along vertical line $x=0.5$.

```{code-cell} ipython3
plt.figure()
ux = u.sub(0).collapse()
ux_vals = ux.x.array[middle_dofs]
y_dofs = Vx.tabulate_dof_coordinates()[middle_dofs, 1]
plt.plot(ux_vals, y_dofs, label="present computation")
Bi = int(float(tau0 / mu))
Bi_data_values = [0, 2, 5, 20]
try:
    i = Bi_data_values.index(Bi)
    data = np.loadtxt(path / "viscoplastic_data.csv", skiprows=1)
    plt.plot(
        data[:, i + 1],
        data[:, 0],
        "o",
        markersize=4,
        label="solution from [Bleyer et al., 2015]",
    )
except ValueError:
    pass
plt.legend()
plt.xlabel("$y$ coordinate")
plt.ylabel("Velocity $u_x$")
plt.show()
```