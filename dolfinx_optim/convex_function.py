#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for convex functions and mesh/facet children.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import numpy as np
from dolfinx import fem
import ufl
import scipy.sparse
import mosek.fusion as mf
from .utils import to_vect, to_list, concatenate
from .cones import RQuad, Quad

MOSEK_CONES = {"quad": mf.Domain.inQCone(), "rquad": mf.Domain.inRotatedQCone()}


def get_shape(expr):
    expr_shape = ufl.shape(expr)
    if len(expr_shape) == 0:  # scalar constraint
        dim = 0
    else:
        dim = expr_shape[0]
    return dim


def petsc_matrix_to_scipy(A_petsc, M_array=None):
    """
    Utility function to converts a PETSc matrix to a scipy sparse coo matrix

    if M_array is not None, computes diag(M_array)^{-1})*A_petsc
    """
    row, col, data = A_petsc.getValuesCSR()
    A_csr = scipy.sparse.csr_matrix((data, col, row), shape=A_petsc.size)
    if M_array is not None:
        A_csr = scipy.sparse.diags(1.0 / M_array) @ A_csr
    return A_csr.tocoo()


def scipy_matrix_to_mosek(A_coo):
    nrow, ncol = A_coo.shape
    return mf.Matrix.sparse(nrow, ncol, A_coo.row, A_coo.col, A_coo.data)


def create_interpolation_matrix(operator, u, V2, dx):
    V1 = u.function_space
    v_ = ufl.TrialFunction(V1)
    v = ufl.TestFunction(V2)
    new_op = ufl.replace(operator, {u: v_})
    a_ufl = ufl.inner(new_op, v) * dx
    A_petsc = fem.petsc.assemble_matrix(fem.form(a_ufl))
    A_petsc.assemble()

    # mass matrix on V2
    one = fem.Function(V2)
    one.vector.set(1.0)
    m_ufl = ufl.inner(one, v) * dx
    M_array = fem.assemble_vector(fem.form(m_ufl)).array

    A_coo = petsc_matrix_to_scipy(A_petsc, M_array=M_array)
    return scipy_matrix_to_mosek(A_coo)


def generate_scalar_quadrature_functionspace(domain, deg_quad):
    We = ufl.FiniteElement(
        "Quadrature", domain.ufl_cell(), degree=deg_quad, quad_scheme="default"
    )
    W = fem.FunctionSpace(domain, We)
    return W


def generate_quadrature_functionspace(domain, deg_quad, shape):
    if shape == () or shape == 0:
        return generate_scalar_quadrature_functionspace(domain, deg_quad)
    try:
        if len(shape) == 1:
            dim = shape[0]
    except TypeError:
        dim = shape
    We = ufl.VectorElement(
        "Quadrature",
        domain.ufl_cell(),
        degree=deg_quad,
        quad_scheme="default",
        dim=dim,
    )
    return fem.FunctionSpace(domain, We)


def _get_mesh_from_expr(expr):
    coeffs = ufl.algorithms.analysis.extract_coefficients(expr)
    for c in coeffs:
        if hasattr(c, "function_space"):
            return c.function_space.mesh
    raise ValueError("Unable to extract mesh from UFL expression")


class ConvexTerm:
    def __init__(self, operand, deg_quad):
        self.domain = _get_mesh_from_expr(operand)
        self.deg_quad = deg_quad
        self.W = generate_scalar_quadrature_functionspace(self.domain, self.deg_quad)
        self.dx = ufl.Measure(
            "dx",
            domain=self.domain,
            metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"},
        )
        self.operand = operand
        self.shape = ufl.shape(self.operand)
        if len(self.shape) == 2:
            self.operand = to_vect(operand)
            self.shape = (len(self.operand),)
        assert self.shape == () or len(self.shape) == 1
        self.W_exp = generate_quadrature_functionspace(
            self.domain, deg_quad, self.shape
        )
        self.ndof = self.W.dofmap.index_map.size_global * self.W.dofmap.index_map_bs
        self.scale_factor = 1.0

        self.variables = []
        self.linear_constraints = []
        self.conic_constraints = []
        self.cones = []
        self.ux = []
        self.lx = []
        self._linear_objective = []

        self.conic_repr(self.operand)

    def add_var(self, dim=0, cone=None, ux=None, lx=None, name=None):
        """
        Add a (list of) auxiliary optimization variable.

        These variables are local and their interpolation is defined through the chosen
        quadrature scheme. They are added to the block structure of the optimization
        problem by following their order of declaration. Inside a ConvexFunction, the
        block structure is :math:`z=[X, Y_0, Y_1, \\ldots, Y_n]` where :math:`X` is the
        global declared variable and each :math:`Y_i` are the additional variables.

        Parameters
        ----------
        dim : int, list of int
            dimension of each variable (0 for a scalar)
        cone : `Cone`, list of `Cone`
            cone in which each variable belongs (None if no constraint)
        ux : float, Function
            upper bound on variable :math:`x\\leq u_x`
        lx : float, Function
            lower bound on variable :math:`x\\leq l_x`
        name : str
            variable name
        """
        dim_list = to_list(dim)
        nlist = len(dim_list)
        new_V_add_var = [
            generate_quadrature_functionspace(self.domain, self.deg_quad, d)
            for d in dim_list
        ]
        self.ux += to_list(ux, nlist)
        self.lx += to_list(lx, nlist)
        if not isinstance(dim, list):
            new_Y = fem.Function(new_V_add_var[0], name=name)
            new_var = new_Y
        else:
            new_Y = [
                fem.Function(v, name=n)
                for (v, n) in zip(new_V_add_var, to_list(name, nlist))
            ]
            new_var = to_list(new_Y, nlist)
        if cone is not None:
            for v, c in zip(new_var, cone):
                self.add_conic_constraint(v, c)
        try:
            self.variables += new_var
            return tuple(new_var)
        except NotImplementedError:
            self.variables.append(new_var)
            return new_var

    def add_eq_constraint(self, Az, b=0, name=None):
        """
        Add an equality constraint :math:`Az=b`.

        `z` can contain a linear combination of X and local variables.

        Parameters
        ----------
        Az : UFL expression
            a UFL linear combination of X and local variables defining the
            linear constraint. We still support expressing Az as a list of
            linear expressions of X and local variable blocks. Use 0 or None for
            an empty block.
        b : float, expression
            corresponding right-hand side
        name : str, optional
            Lagrange-multiplier name for later retrieval.
        """
        self.add_ineq_constraint(Az, b, b, name)

    def add_ineq_constraint(self, Az, bu=None, bl=None, name=None):
        """
        Add an inequality constraint :math:`b_l \\leq Az \\leq b_u`.

        `z` can contain a linear combination of X and local variables.

        Parameters
        ----------
        Az : UFL expression
            a UFL linear combination of X and local variables defining the
            linear constraint. We still support expressing Az as a list of
            linear expressions of X and local variable blocks. Use 0 or None for
            an empty block.
        b_l : float, expression
            corresponding lower bound. Ignored if None.
        b_u : float, expression
            corresponding upper bound. Ignored if None
        name : str, optional
            Lagrange-multiplier name for later retrieval.
        """
        A_shape = ufl.shape(Az)
        if len(A_shape) == 0:  # scalar constraint
            dim = 0
        else:
            dim = A_shape[0]
        self.linear_constraints.append(
            {
                "A": Az,
                "bu": bu,
                "bl": bl,
                "dim": dim,
                "V": generate_quadrature_functionspace(self.domain, self.deg_quad, dim),
                "name": name,
            }
        )

    def add_conic_constraint(self, expr, cone, name=""):
        dim = get_shape(expr)
        assert (
            cone.dim == dim
        ), f"Expression and cone dimensions do not match: {dim} vs. {cone.dim}."
        self.conic_constraints.append(
            {
                "expr": expr,
                "cone": cone,
                "dim": dim,
                "V": generate_quadrature_functionspace(self.domain, self.deg_quad, dim),
                "name": name,
            }
        )

    def add_linear_term(self, expr):
        """
        Add a linear combination term of X and local variables.

        Parameters
        ----------
        expr : UFL expression
            a UFL linear combination of X and local variables defining the
            linear objective.
        """
        self._linear_objective.append(expr * self.dx)

    @property
    def objective(self):
        return sum(self._linear_objective)
        c = fem.assemble_vector(
            fem.form(self.scale_factor * ufl.TestFunction(self.W) * self.dx)
        ).array
        try:
            return mf.Expr.dot(c, self.obj_vector)
        except AttributeError:
            return mf.Expr.constTerm(0.0)

    def _apply_objective(self, problem):
        obj = []
        for var, vec in zip(problem.variables, problem.vectors):
            var_ = ufl.TestFunction(var.function_space)
            dobj = ufl.derivative(self.scale_factor * self.objective, var, var_)
            c = fem.assemble_vector(fem.form(dobj)).array
            try:
                obj.append(mf.Expr.dot(c, vec))
            except AttributeError:
                obj.append(mf.Expr.constTerm(0.0))
        return mf.Expr.add(obj)

    def _apply_linear_constraints(self, problem):
        for cons in self.linear_constraints:
            expr = cons["expr"]
            expr_list = []
            lamb_ = ufl.TestFunction(cons["V"])
            # try:
            for var, vec in zip(problem.variables, problem.vectors):
                if isinstance(var, fem.Function):
                    dvar = ufl.TrialFunction(var.function_space)
                    curr_expr = ufl.derivative(expr, var, dvar)
                    A_petsc = fem.petsc.assemble_matrix(
                        fem.form(ufl.inner(lamb_, curr_expr) * self.dx)
                    )
                    A_petsc.assemble()
                    A_coo = petsc_matrix_to_scipy(A_petsc)
                    A_mosek = scipy_matrix_to_mosek(A_coo)
                    expr_list.append(mf.Expr.mul(A_mosek, vec))

            if hasattr(cons, "bu") and hasattr(cons, "bl"):
                bu = fem.assemble_vector(
                    fem.form(ufl.inner(lamb_, cons["bu"]) * self.dx)
                )
                bl = fem.assemble_vector(
                    fem.form(ufl.inner(lamb_, cons["bl"]) * self.dx)
                )
                problem.M.constraint(
                    mf.Expr.add(expr_list), mf.Domain.inRange(xbl, xbu)
                )
            else:
                problem.M.constraint(mf.Expr.add(expr_list), mf.Domain.equalsTo(0.0))

    def _apply_conic_constraints(self, problem):
        print(self.conic_constraints)
        for cons in self.conic_constraints:
            expr = cons["expr"]
            cone = cons["cone"]
            V = cons["V"]

            print("Conic constraint:", cone)

            v = ufl.TestFunction(V)
            expr_list = []
            curr_expr_list = []
            for var, vec in zip(problem.variables, problem.vectors):
                curr_expr = ufl.derivative(expr, var, var)
                curr_expr_list.append(curr_expr)
                A_op = create_interpolation_matrix(curr_expr, var, V, self.dx)
                expr_list.append(mf.Expr.mul(A_op, vec))

            b = expr - sum(curr_expr_list)
            b_vec = mf.Expr.constTerm(
                fem.assemble_vector(fem.form(ufl.inner(b, v) * self.dx)).array
            )
            expr_list.append(b_vec)
            z_in_cone = mf.Expr.add(expr_list)
            if len(self.shape) > 0:
                print(z_in_cone.getSize())
                print(self.shape)
                print(self.ndof)
                z_shape = get_shape(expr)
                assert (
                    z_in_cone.getSize() == self.ndof * z_shape
                ), "Wrong shape in conic constraint"
                z_in_cone = mf.Expr.reshape(z_in_cone, self.ndof, z_shape)

            problem.M.constraint(z_in_cone, MOSEK_CONES[cone.type])

    def _apply_on_problem(self, problem):
        expr_list = []
        for var, vec in zip(problem.variables, problem.vectors):
            if isinstance(var, fem.Function):
                curr_operand = ufl.derivative(self.operand, var, var)
                try:
                    A_op = create_interpolation_matrix(
                        curr_operand, var, self.W_exp, self.dx
                    )
                    expr_list.append(mf.Expr.mul(A_op, vec))
                except:
                    print("Empty block, skipping")
        self.expr = mf.Expr.add(expr_list)
        if len(self.shape) > 0:
            self.expr = mf.Expr.reshape(self.expr, self.ndof, self.shape[0])

    def __rmul__(self, alpha):
        """Allow multiplication by a scalar."""
        if type(alpha) in [float, int, fem.Constant] or isinstance(
            alpha, ufl.core.expr.Expr
        ):
            self.scale_factor = alpha
        return self

    # def add_var(self, M, m=0, name=None):
    #     if m <= 1:
    #         var = M.variable(name, self.ndof)
    #     else:
    #         var = M.variable(name, (m, self.ndof))
    #     self.variables.append(var)
    #     return var


# class Variable:
#     def __init__(self, name, dim=0):
#         self.name = name
#         self.dim = dim


# class ConstantTerm:
#     def __init__(self, value):
#         self.value = value


# class AffineExpression:
#     def __init__(self, expr):
#         self.expr = expr

#     def get_linear_term(self):
#         return

#     def get_constant_term(self):
#         return


# expr = grad(u)
# t = self.add_var(name)
# stack = ufl.as_vector([1, t, expr])

# self.add_conic_constraint(stack, Quad)
# self.add_eq_constraint(t, b)
# self.add_ineq_constraint(expr, bl, bu)


# self._apply_on_problem()


class QuadraticTerm(ConvexTerm):
    def conic_repr(self, expr):
        t = self.add_var()
        stack = concatenate([1.0 / 2.0, t, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, RQuad(dim))
        self.add_linear_term(t)


class L2Norm(ConvexTerm):
    def conic_repr(self, expr):
        t = self.add_var(name="t")
        stack = concatenate([t, expr])
        dim = get_shape(stack)
        self.add_conic_constraint(stack, Quad(dim))
        self.add_linear_term(t)


# class QuadraticTerm(ConvexTerm):
#     def conic_repr(self, M):
#         t = M.variable("obj-" + self.__class__.__nam)e__, self.ndof)
#         # s = M.variable("s", self.ndof, mf.Domain.equalsTo(1.0 / 2.0))
#         stack = mf.Expr.hstack([mf.Expr.constTerm(self.ndof, 1.0 / 2.0), t, self.expr])
#         M.constraint(stack, mf.Domain.inRotatedQCone())
#         self.obj_vector = t


# class L2Norm(ConvexTerm):
#     def conic_repr(self, M):
#         t = M.variable("obj-" + self.__class__.__name__, self.ndof)
#         stack = mf.Expr.hstack([t, self.expr])
#         M.constraint(stack, mf.Domain.inQCone())
#         self.obj_vector = t


class L2Ball(ConvexTerm):
    def conic_repr(self, M):
        stack = mf.Expr.hstack([mf.Expr.constTerm(self.ndof, 1.0), self.expr])
        M.constraint(stack, mf.Domain.inQCone())
