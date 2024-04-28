import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl
from ufl import dot, grad
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import ConvexTerm


d = 1


class LinearEquality(ConvexTerm):
    def conic_repr(self, expr):
        self.add_var(d)
        self.add_eq_constraint(expr - 1.0)


domain = mesh.create_unit_interval(
    MPI.COMM_WORLD,
    1,
)

V = fem.functionspace(domain, ("P", 1))
prob = MosekProblem(domain, "test")
u = prob.add_var(V)
fun = LinearEquality(u, 1)
prob.add_convex_term(fun)
print(dir(fun))
print(fun.linear_constraints)
