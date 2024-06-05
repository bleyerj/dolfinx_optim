import ufl
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl.algorithms
import ufl.algorithms.apply_algebra_lowering
from dolfinx_optim.utils import split_affine_expression, concatenate

N = 1
domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    N,
    N,
    cell_type=mesh.CellType.quadrilateral,
)

V = fem.functionspace(domain, ("P", 1))
V0 = fem.functionspace(domain, ("DG", 0, (2,)))

u = fem.Function(V, name="u")
g = fem.Function(V0, name="g")
e = ufl.as_vector([1.0, 1.0])
op = ufl.variable(ufl.grad(u))
expr = 2 * op - 3 * g - e
Au, Ag, c = split_affine_expression(expr, op, [g])
