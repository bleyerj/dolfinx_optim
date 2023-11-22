#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mosek_io provides interface to the Mosek optimization solver.

All types of conic programming (LP, SOCP, SDP, power and exponential cones)
are supported.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import numpy as np
import sys
import mosek
import mosek.fusion as mf
import ufl
from ufl.algorithms.apply_derivatives import apply_derivatives
import scipy.sparse
from dolfinx import fem
from .utils import get_shape


def to_list(a, n=1):
    """Transform `a` to list of length `n`."""
    if type(a) not in [list, tuple]:
        return [a] * n
    else:
        return a


def mosek_cone_domain(K):
    if K.type == "quad":
        return mf.Domain.inQCone()
    elif K.type == "rquad":
        return mf.Domain.inRotatedQCone()
    elif K.type == "ppow":
        return mf.Domain.inPPowerCone(K.alpha)
    elif K.type == "pexp":
        return mf.Domain.inPExpCone()
    elif K.type == "dpow":
        return mf.Domain.inDPowerCone(K.alpha)
    elif K.type == "dexp":
        return mf.Domain.inDExpCone()
    elif K.type == "sdp":
        return mf.Domain.inPSDCone()
    else:
        raise NotImplementedError(f'Cone type "{K.type}" is not available.')


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


class MosekProblem:
    """A generic optimization problem using the Mosek optimization solver."""

    def __init__(self, domain, name):
        self.name = name
        self.Vx = []
        self.Vy = []
        self.lagrange_multiplier_names = []
        self.cones = []
        self.ux = []
        self.lx = []
        self.variables = []
        self.variable_names = []
        self.int_var = []
        self.A = []
        self.bu = []
        self.bl = []
        self.bc_dual = []
        self.bc_prim = []
        self.c = []
        self.parameters = self._default_parameters()
        self.M = mf.Model(name)
        # self.vectors_dict = {}
        self.vectors = []
        self.objectives = []
        self.constraints = {}
        self.domain = domain

    def _default_parameters(self):
        return {
            "presolve": "free",
            "presolve_lindep": "off",
            "log_level": 10,
            "tol_rel_gap": 1e-7,
            "solve_form": "free",
            "num_threads": 0,
            "dump_file": None,
        }

    def _create_variable_vector(self, var, name, ux, lx):
        if ux is None and lx is None:
            domain = mf.Domain.unbounded()
        elif ux is None:
            domain = mf.Domain.greaterThan(lx)
        elif lx is None:
            domain = mf.Domain.greaterThan(ux)
        else:
            domain = mf.Domain.inRange(lx, ux)
        if isinstance(var, fem.Function):
            if name is None:
                return self.M.variable(len(var.vector.array), domain)
            else:
                return self.M.variable(name, len(var.vector.array), domain)
        else:
            value = var.value
            if len(value.shape) == 0:
                size = 1
            else:
                size = len(value)
            if name is None:
                return self.M.variable(size, domain)
            else:
                return self.M.variable(name, size, domain)

    def _add_boundary_conditions(self, variable, vector, bcs):
        V = variable.function_space
        u_bc = fem.Function(V)
        u_bc.vector.set(np.inf)
        fem.set_bc(u_bc.vector, to_list(bcs))
        dof_indices = np.where(u_bc.vector.array < np.inf)[0].astype(np.int32)
        bc_values = u_bc.vector.array[dof_indices]
        self.M.constraint(vector.pick(dof_indices), mf.Domain.equalsTo(bc_values))

    def add_var(
        self,
        V,
        cone=None,
        lx=None,
        ux=None,
        bc=None,
        name=None,
        int_var=False,
    ):
        """Add a (list of) optimization variable.

        The added variables belong to the corresponding FunctionSpace V.

        Parameters
        ----------
        V : (list of) `FunctionSpace`
            variable FunctionSpace
        cone : (list of) `Cone`
            cone in which each variable belongs (None if no constraint)
        ux : (list of) float, Function
            upper bound on variable :math:`x \\leq u_x`
        lx : (list of) float, Function
            lower bound on variable :math:`l_x \\leq x`
        bc : (list of) `DirichletBC`
            boundary conditions applied to the variables (None if no bcs)
        name : (list of) str
            name of the associated functions
        int_var : (list of) bool
            True if variable is an integer, False if it is continuous (default)

        Returns
        -------
        x : Function tuple
            optimization variables
        """
        if not isinstance(V, list):
            V_list = [V]
        else:
            V_list = V
        nlist = len(V_list)
        bc_list = to_list(bc, nlist)

        self.lx += to_list(lx, nlist)
        self.ux += to_list(ux, nlist)
        self.cones += to_list(cone, nlist)
        self.bc_prim += to_list(bc_list, nlist)
        self.variable_names = to_list(name, nlist)
        self.int_var += to_list(int_var, nlist)

        new_var = []
        for V, name in zip(V_list, self.variable_names):
            if isinstance(V, fem.FunctionSpace):
                new_var.append(fem.Function(V, name=name))
            elif type(V) == int:
                if V <= 1:
                    value = 0.0
                else:
                    value = np.zeros((V,))
                new_var.append(fem.Constant(self.domain, value))

        for var, bc, name, ux, lx in zip(
            new_var, bc_list, self.variable_names, self.ux, self.lx
        ):
            vector = self._create_variable_vector(
                var, name, ux, lx
            )  # FIXME: better handle bcs ?
            if bc is not None:
                self._add_boundary_conditions(var, vector, bc)
            self.vectors.append(vector)
        self.variables += new_var
        self.Vx += V_list

        if nlist == 1:
            return new_var[0]

        return tuple(new_var)

    def add_eq_constraint(self, A=None, b=0.0, bc=None, name=None):
        """
        Add a linear equality constraint :math:`Ax = b`.

        The constraint matrix A is expressed through a bilinear form involving
        the corresponding Lagrange multiplier defined on the space Vy.
        The right-hand side is a linear form involving the same Lagrange multiplier.
        Naming the constraint enables to retrieve the corresponding Lagrange multiplier
        optimal value.

        Parameters
        ----------
        Vy : `FunctionSpace`
            FunctionSpace of the corresponding Lagrange multiplier
        A : function
            A function of signature `y -> bilinear_form` where the function
            argument `y` is the constraint Lagrange multiplier.
        b : float, function
            A float or a function of signature `y -> linear_form` where the function
            argument `y` is the constraint Lagrange multiplier (default is 0.0)
        bc : DirichletBC
            boundary conditions to apply on the Lagrange multiplier (will be
            applied to all columns of the constraint when possible)
        name : str
            Lagrange multiplier name
        """
        self.add_ineq_constraint(A, b, b, bc, name)

    def add_ineq_constraint(self, A=None, bu=None, bl=None, bc=None, name=None):
        """
        Add a linear inequality constraint :math:`b_l \\leq Ax \\leq b_u`.

        The constraint matrix A is expressed through a bilinear form involving
        the corresponding Lagrange multiplier defined on the space Vy.
        The right-hand sides are linear forms involving the same Lagrange multiplier.
        Naming the constraint enables to retrieve the corresponding Lagrange multiplier
        optimal value.

        Parameters
        ----------
        Vy : `FunctionSpace`
            FunctionSpace of the corresponding Lagrange multiplier
        A : function
            A function of signature `y -> bilinear_form` where the function
            argument `y` is the constraint Lagrange multiplier.
        bl : float, function
            A float or a function of signature `y -> linear_form` where the function
            argument `y` is the constraint Lagrange multiplier (default is 0.0)
        bu : float, function
            same as bl
        bc : DirichletBC
            boundary conditions to apply on the Lagrange multiplier (will be
            applied to all columns of the constraint when possible)
        name : str
            Lagrange multiplier name
        """
        expr = []
        arity = len(A.arguments())
        if arity == 0:
            for var, vec in zip(self.variables, self.vectors):
                if isinstance(var, fem.Function):
                    var_ = ufl.TestFunction(var.function_space)
                    curr_form = ufl.derivative(A, var, var_)
                    c = fem.assemble_vector(fem.form(curr_form)).array
                    expr.append(mf.Expr.dot(c, vec))
            self.M.constraint(mf.Expr.add(expr), mf.Domain.inRange(bl, bu))

        elif arity == 1:
            ufl_element = A.arguments()[0].ufl_function_space().ufl_element()
            V_cons = fem.FunctionSpace(self.domain, ufl_element)

            for var, vec in zip(self.variables, self.vectors):
                if isinstance(var, fem.Function):
                    dvar = ufl.TrialFunction(var.function_space)
                    curr_form = ufl.derivative(A, var, dvar)
                    try:
                        # FIXME: need to apply bcs
                        A_petsc = fem.petsc.assemble_matrix(fem.form(curr_form))
                        A_petsc.assemble()
                        A_coo = petsc_matrix_to_scipy(A_petsc)
                        A_mosek = scipy_matrix_to_mosek(A_coo)
                        expr.append(mf.Expr.mul(A_mosek, vec))
                    except:
                        print("Empty block, skipping")
                else:
                    n = vec.getSize()
                    c0 = fem.assemble_vector(fem.form(A)).array
                    c = []
                    for val in np.eye(n):
                        var.value = val
                        c.append(fem.assemble_vector(fem.form(A)).array - c0)
                        var.value *= 0
                    c_mat = mf.Matrix.dense(c).transpose()
                    expr.append(mf.Expr.mul(c_mat, vec))

            expr_tot = mf.Expr.add(expr)
            if type(bl) in [float, int]:
                xbl = np.full(expr_tot.getSize(), bl)
            else:
                xbl = fem.assemble_vector(fem.form(bl))
            if type(bu) in [float, int]:
                xbu = np.full(expr_tot.getSize(), bu)
            else:
                xbu = fem.assemble_vector(fem.form(bu))
            constraint = self.M.constraint(expr_tot, mf.Domain.inRange(xbl, xbu))
            self.constraints.update({name: (constraint, V_cons)})

    def add_obj_func(self, obj):
        """
        Add an objective function.

        Parameters
        ----------
        obj : linear form
        """
        for var, vec in zip(self.variables, self.vectors):
            if isinstance(var, fem.Function):
                var_ = ufl.TestFunction(var.function_space)
                curr_form = apply_derivatives(ufl.derivative(obj, var, var_))
                if len(curr_form.arguments()) > 0:
                    c = fem.assemble_vector(fem.form(curr_form)).array
                    self.objectives.append(mf.Expr.dot(c, vec))
            else:
                n = vec.getSize()
                c0 = fem.assemble_scalar(fem.form(obj))
                c = []
                for val in np.eye(n):
                    var.value = val
                    c.append(fem.assemble_scalar(fem.form(obj)) - c0)
                self.objectives.append(mf.Expr.dot(np.array(c), vec))

    def add_convex_term(self, conv_fun):
        """Add the convex term `conv_fun` to the problem."""
        conv_fun._apply_conic_representation()

        for var, name, ux, lx in zip(
            conv_fun.variables, conv_fun.variable_names, conv_fun.ux, conv_fun.lx
        ):
            vector = self._create_variable_vector(var, name, ux, lx)
            self.variables.append(var)
            self.vectors.append(vector)

        objective = self._apply_objective(conv_fun)
        self._apply_linear_constraints(conv_fun)
        self._apply_conic_constraints(conv_fun)
        if objective is not None:
            self.objectives.append(objective)

    def _apply_objective(self, conv_fun):
        obj = []
        for var, vec in zip(self.variables, self.vectors):
            if conv_fun.objective is not None:
                if isinstance(var, fem.Constant):
                    if np.ndim(var.value) == 0:
                        var.value = 1.0
                        c = fem.assemble_scalar(
                            fem.form(conv_fun.scale_factor * conv_fun.objective)
                        )
                    else:
                        c = np.zeros_like(var.value)
                        for i in range(len(c)):
                            var.value *= 0.0
                            var.value[i] = 1.0
                            c[i] = fem.assemble_scalar(
                                fem.form(conv_fun.scale_factor * conv_fun.objective)
                            )
                else:
                    print("Objective", conv_fun.objective)
                    var_ = ufl.TestFunction(var.function_space)
                    dobj = ufl.derivative(
                        conv_fun.scale_factor * conv_fun.objective, var, var_
                    )
                    c = fem.assemble_vector(fem.form(dobj)).array
                try:
                    obj.append(mf.Expr.dot(c, vec))
                except AttributeError:
                    obj.append(mf.Expr.constTerm(0.0))
        if len(obj) > 0:
            return mf.Expr.add(obj)
        else:
            return None

    def _apply_linear_constraints(self, conv_fun):
        for cons in conv_fun.linear_constraints:
            expr = cons["expr"]
            expr_list = []
            curr_expr_list = []
            lamb_ = ufl.TestFunction(cons["V"])

            for var, vec in zip(self.variables, self.vectors):
                if not isinstance(var, fem.Constant):
                    curr_expr = apply_derivatives(ufl.derivative(expr, var, var))

                    if not curr_expr == 0:  # if derivative is zero ignore
                        curr_expr_list.append(curr_expr)
                        A_op = create_interpolation_matrix(
                            curr_expr, var, cons["V"], conv_fun.dx
                        )
                        expr_list.append(mf.Expr.mul(A_op, vec))

                else:
                    pass  # FIXME: handle constant terms
                    # raise NotImplementedError

            if cons["bu"] is not None:
                if isinstance(cons["bu"], float):
                    bu = fem.Function(cons["V"])
                    xbu = bu.vector.array
                    xbu[:] = cons["bu"]
                elif cons["bu"] == 0:
                    bu = fem.Function(cons["V"])
                    xbu = bu.vector.array
                else:
                    bu = fem.assemble_vector(
                        fem.form(ufl.inner(lamb_, cons["bu"]) * conv_fun.dx)
                    )
                    xbu = bu.array
            else:
                xbu = None
            if cons["bl"] is not None:
                if isinstance(cons["bl"], float):
                    bl = fem.Function(cons["V"])
                    xbl = bl.vector.array
                    xbl[:] = cons["bl"]
                elif cons["bl"] == 0:
                    bl = fem.Function(cons["V"])
                    xbl = bl.vector.array
                else:
                    bl = fem.assemble_vector(
                        fem.form(ufl.inner(lamb_, cons["bl"]) * conv_fun.dx)
                    )
                    xbl = bl.array
            else:
                xbl = None

            if (xbu is not None) and (xbl is not None):
                self.M.constraint(mf.Expr.add(expr_list), mf.Domain.inRange(xbl, xbu))
            elif xbu is not None:
                self.M.constraint(mf.Expr.add(expr_list), mf.Domain.lessThan(xbu))
            elif xbl is not None:
                self.M.constraint(mf.Expr.add(expr_list), mf.Domain.greaterThan(xbl))
            else:
                raise NotImplementedError(
                    "Linear inequality constraint must have bounds"
                )

    def _apply_conic_constraints(self, conv_fun):
        for cons in conv_fun.conic_constraints:
            expr = cons["expr"]
            cone = cons["cone"]
            V = cons["V"]

            v = ufl.TestFunction(V)
            expr_list = []
            curr_expr_list = []
            for var, vec in zip(self.variables, self.vectors):
                if not isinstance(var, fem.Constant):
                    print("var conic", var)
                    curr_expr = apply_derivatives(ufl.derivative(expr, var, var))
                    if not curr_expr == 0:  # if derivative is zero ignore
                        curr_expr_list.append(curr_expr)
                        A_op = create_interpolation_matrix(
                            curr_expr, var, V, conv_fun.dx
                        )
                        expr_list.append(mf.Expr.mul(A_op, vec))
                else:
                    pass  # FIXME: handle constant terms
                    # raise NotImplementedError

            if len(curr_expr_list) > 0:
                b = expr - sum(curr_expr_list)
            else:
                b = expr
            b_vec = mf.Expr.constTerm(
                fem.assemble_vector(fem.form(ufl.inner(b, v) * conv_fun.dx)).array
            )
            expr_list.append(b_vec)
            z_in_cone = mf.Expr.add(expr_list)
            z_shape = get_shape(expr)
            if cone.type == "sdp":
                d = cone.dim
                print(z_in_cone.getSize(), conv_fun.ndof, d)
                assert z_in_cone.getSize() == conv_fun.ndof * d * d
                z_in_cone = mf.Expr.reshape(z_in_cone, [conv_fun.ndof, d, d])
            elif z_shape > 0:
                assert (
                    z_in_cone.getSize() == conv_fun.ndof * z_shape
                ), "Wrong shape in conic constraint"
                z_in_cone = mf.Expr.reshape(z_in_cone, conv_fun.ndof, z_shape)
            else:
                z_in_cone = mf.Expr.reshape(z_in_cone, conv_fun.ndof, 1)

            self.M.constraint(z_in_cone, mosek_cone_domain(cone))

    def _set_task_parameters(self):
        assert all(
            [p in self._default_parameters().keys() for p in self.parameters.keys()]
        ), "Available parameters are:\n{}".format(self._default_parameters())

        self.M.setSolverParam("autoUpdateSolInfo", "on")

        # Set log level (integer parameter)
        self.M.setSolverParam("log", self.parameters["log_level"])
        # Select interior-point optimizer... (parameter with symbolic string values)
        self.M.setSolverParam("optimizer", "conic")
        # ... without basis identification (parameter with symbolic string values)
        self.M.setSolverParam("intpntBasis", "never")

        # ... without basis identification (parameter with symbolic string values)
        self.M.setSolverParam("presolveUse", self.parameters["presolve"])
        self.M.setSolverParam("presolveLindepUse", self.parameters["presolve_lindep"])

        # Set relative gap tolerance (double parameter)
        self.M.setSolverParam("intpntCoTolRelGap", self.parameters["tol_rel_gap"])
        # Controls whether primal or dual form is solved
        self.M.setSolverParam("intpntSolveForm", self.parameters["solve_form"])

    def get_solution_info(self, output=True):
        """
        Return information dictionary on the solution.

        If output=True, it gets printed out.
        """
        int_info = [
            "intpntIter",
            "optNumcon",
            "optNumvar",
            "anaProNumVar",
        ]
        double_info = [
            "optimizerTime",
            "presolveEliTime",
            "presolveLindepTime",
            "presolveTime",
            "intpntTime",
            "intpntOrderTime",
            "solItrPrimalObj",
            "solItrDualObj",
        ]
        Lint_info = ["numAffConicCon"]
        Lint_value = [self.M.getSolverLIntInfo("rdNumacc")]
        int_value = [self.M.getSolverIntInfo(k) for k in int_info] + Lint_value
        double_value = [self.M.getSolverDoubleInfo(k) for k in double_info]
        info = dict(
            zip(
                int_info + Lint_info + double_info,
                int_value + Lint_value + double_value,
            )
        )
        info.update(
            {"solution_status": str(self.M.getAcceptedSolutionStatus()).split(".")[1]}
        )
        if output:
            print("Solver information:\n{}".format(info))
        return info

    def optimize(self, sense="min", get_bound_dual=False):
        """
        Write the problem in Mosek format and solves.

        Parameters
        ----------
        sense : {"min[imize]", "max[imize]"}
            sense of optimization
        get_bound_dual : bool
            if True, optimal dual variable bounds will be stored in `self.sux` and
            `self.slx`

        Returns
        -------
        pobj : float
            the computed optimal value
        """
        if sense in ["minimize", "min"]:
            self.sense = mf.ObjectiveSense.Minimize
        elif sense in ["maximize", "max"]:
            self.sense = mf.ObjectiveSense.Maximize

        if len(self.objectives) == 0:
            raise ValueError("No objective function has been defined.")
        self.M.objective(self.sense, mf.Expr.add(self.objectives))
        self.M.writeTask("dump.ptf")
        self.M.setLogHandler(sys.stdout)
        self._set_task_parameters()
        self.M.solve()

        # Retrieve solution and save to file
        for var, vec in zip(self.variables, self.vectors):
            if isinstance(var, fem.Function):
                var.vector.array[:] = vec.level()
            else:
                var.value = vec.level()

        return self.M.primalObjValue(), self.M.dualObjValue()

    def get_lagrange_multiplier(self, name):
        constraint, V_cons = self.constraints[name]
        lag = fem.Function(V_cons, name=name)
        lag.vector.array[:] = constraint.dual()
        return lag
