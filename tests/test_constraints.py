import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl
from ufl import dot, grad
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import ConvexTerm
from dolfinx_optim.utils import get_shape


d = 1


class LinearEquality(ConvexTerm):
    def conic_repr(self, expr):
        self.add_eq_constraint(expr, b=1.0)


class LinearInequality(ConvexTerm):
    def conic_repr(self, expr):
        self.add_ineq_constraint(expr, bu=1.0, bl=1.0)


class LinearEquality2(ConvexTerm):
    def conic_repr(self, expr):
        d = get_shape(expr)
        if d == 0:
            b = 2.0
        else:
            b = ufl.as_vector([2.0] * d)
        self.add_eq_constraint(2 * expr - b)


constraints = [LinearEquality, LinearEquality2, LinearInequality]
dimensions = [0, 1, 2]


@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("cons", constraints)
def test_linear_equality(dim, cons):
    domain = mesh.create_unit_interval(
        MPI.COMM_WORLD,
        5,
    )
    if dim == 0:
        shape = ()
    else:
        shape = (dim,)
    V = fem.functionspace(domain, ("DG", 0, shape))
    prob = MosekProblem(domain, "test")
    u = prob.add_var(V)
    fun = cons(u, 2)
    prob.add_convex_term(fun)
    prob.optimize()
    assert np.allclose(u.x.array, 1.0)
