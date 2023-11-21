#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for convex functions and mesh/facet children.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
from abc import ABC, abstractmethod
from dolfinx import fem
import ufl
import mosek.fusion as mf
from dolfinx_optim.utils import to_vect, to_list, get_shape

MOSEK_CONES = {"quad": mf.Domain.inQCone(), "rquad": mf.Domain.inRotatedQCone()}


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


class ConvexTerm(ABC):
    def __init__(self, operand, deg_quad, parameters=None):
        self.domain = _get_mesh_from_expr(operand)
        self.deg_quad = deg_quad
        self.W = generate_scalar_quadrature_functionspace(self.domain, self.deg_quad)
        self.dx = ufl.Measure(
            "dx",
            domain=self.domain,
            metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"},
        )

        self.operand_shape = ufl.shape(operand)
        if len(self.operand_shape) == 2:
            operand = to_vect(operand)
            self.operand_shape = (len(operand),)
        assert self.operand_shape == () or len(self.operand_shape) == 1
        self.operand = ufl.variable(
            operand
        )  # use variable to differentiate expressions

        self.W_exp = generate_quadrature_functionspace(
            self.domain, deg_quad, self.operand_shape
        )
        self.ndof = self.W.dofmap.index_map.size_global * self.W.dofmap.index_map_bs
        self.scale_factor = 1.0
        self.parameters = parameters

        self.variables = []
        self.variable_names = []
        self.linear_constraints = []
        self.conic_constraints = []
        self.ux = []
        self.lx = []
        self._linear_objective = []

    def _apply_conic_representation(self):
        if self.parameters is None:
            self.conic_repr(self.operand)
        else:
            self.conic_repr(self.operand, *self.parameters)

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
        self.variable_names += to_list(name, nlist)
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
        if isinstance(new_var, list):
            self.variables += new_var
            return tuple(new_var)
        else:
            self.variables.append(new_var)
            return new_var

    def add_eq_constraint(self, Az, b=0.0, name=None):
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
        if isinstance(b, int):
            b = float(b)
        self.add_ineq_constraint(Az, b, b, name)

    def add_ineq_constraint(self, expr, bu=None, bl=None, name=None):
        """
        Add an inequality constraint :math:`b_l \\leq expr \\leq b_u`.

        Parameters
        ----------
        expr : UFL expression
            a UFL affine combination of variables
        b_l : float, expression
            corresponding lower bound. Ignored if None.
        b_u : float, expression
            corresponding upper bound. Ignored if None
        name : str, optional
            Lagrange-multiplier name for later retrieval.
        """
        if isinstance(bu, int):
            bu = float(bu)
        if isinstance(bl, int):
            bl = float(bl)
        dim = get_shape(expr)
        # _, _, constant = split_affine_expression(expr, self.operand, self.variables)
        # print(get_shape(constant), constant)
        # if bu is not None:
        #     d1 = get_shape(bu)
        #     d2 = get_shape(constant)
        #     if d1 != d2:
        #         assert d1 == 0, "Rhs should be a constant, shape mismatch!"
        #         bu = reshape(bu, d2)
        #     bu -= constant
        # if bl is not None:
        #     d1 = get_shape(bl)
        #     d2 = get_shape(constant)
        #     if d1 != d2:
        #         assert d1 == 0, "Rhs should be a constant, shape mismatch!"
        #         bl = reshape(bl, d2)
        #     bl -= constant
        self.linear_constraints.append(
            {
                "expr": expr,
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
        self._linear_objective.append(expr)

    @property
    def objective(self):
        if len(self._linear_objective) == 0:
            return None
        else:
            return sum(self._linear_objective) * self.dx

    def __rmul__(self, alpha):
        """Allow multiplication by a scalar."""
        if type(alpha) in [float, int, fem.Constant] or isinstance(
            alpha, ufl.core.expr.Expr
        ):
            self.scale_factor = alpha
        return self

    def copy(self, fun):
        self.linear_constraints = fun.linear_constraints
        self.conic_constraints = fun.conic_constraints
        self.variables = fun.variables
        self.variable_names = fun.variable_names
        self.scale_factor = fun.scale_factor
        self.ux = fun.ux
        self.lx = fun.lx
        self.parameters = fun.parameters
        self._linear_objective = fun._linear_objective

    @abstractmethod
    def conic_repr(self, expr):
        pass
