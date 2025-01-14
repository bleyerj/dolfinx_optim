import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import ConvexTerm

domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)

V = fem.functionspace(domain, ("P", 1))


def test_objective():
    prob = MosekProblem(domain, "test")
    x = prob.add_var(
        V,
        lx=1.0,
    )
    prob.add_obj_func(3 * x * ufl.dx)
    pobj, dobj = prob.optimize()
    assert np.allclose(x.x.array, 1.0)
    assert np.isclose(pobj, 3.0) and np.isclose(dobj, 3.0)


def test_convex_term():
    class LinearTerm(ConvexTerm):
        def conic_repr(self, x):
            self.add_linear_term(3 * x)

    prob = MosekProblem(domain, "test")
    x = prob.add_var(
        V,
        lx=1.0,
    )
    print(prob.variables[0] is x)
    prob.add_convex_term(LinearTerm(x, 1))
    pobj, dobj = prob.optimize(dump=True)
    assert np.allclose(x.x.array, 1.0)
    assert np.isclose(pobj, 3.0) and np.isclose(dobj, 3.0)


test_convex_term()
