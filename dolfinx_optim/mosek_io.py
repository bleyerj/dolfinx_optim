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
import scipy.sparse as sp
import numpy as np
import sys
import mosek
import mosek.fusion as mf
import ufl
from itertools import compress
from dolfinx import fem
from .convex_function import scipy_matrix_to_mosek, petsc_matrix_to_scipy


def to_list(a, n=1):
    """Transform `a` to list of length `n`."""
    if type(a) not in [list, tuple]:
        return [a] * n
    else:
        return a


# unimportant value to denote infinite bounds
inf = 1e30

MOSEK_CONE_TYPES = {"quad": mosek.conetype.quad, "rquad": mosek.conetype.rquad}
version = mosek.Env().getversion()
if version >= (9, 0, 0):
    MOSEK_CONE_TYPES.update(
        {
            "ppow": mosek.conetype.ppow,
            "dpow": mosek.conetype.dpow,
            "pexp": mosek.conetype.pexp,
            "dexp": mosek.conetype.dexp,
        }
    )


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
        self.vectors_dict = {}
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

    def _create_variable_vector(self, var, name):
        if name is None:
            name = ""
        if isinstance(var, fem.Function):
            return self.M.variable(name, len(var.vector.array))
        else:
            value = var.value
            if len(value.shape) == 0:
                return self.M.variable(name, 1)
            else:
                return self.M.variable(name, len(value))

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

        for var, bc, name in zip(new_var, bc_list, self.variable_names):
            vector = self._create_variable_vector(
                var, name
            )  # FIXME: better handle bcs ?
            print(name, vector, vector.getSize())
            if bc is not None:
                self._add_boundary_conditions(var, vector, bc)
            self.vectors_dict.update({name: vector})
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
                curr_form = ufl.derivative(obj, var, var_)
                c = fem.assemble_vector(fem.form(curr_form)).array
                self.objectives.append(mf.Expr.dot(c, vec))
            else:
                n = vec.getSize()
                c0 = fem.assemble_scalar(fem.form(obj))
                c = []
                for val in np.eye(n):
                    var.value = val
                    c.append(fem.assemble_scalar(fem.form(obj)) - c0)
                c = np.array([1.0])
                self.objectives.append(mf.Expr.dot(np.array(c), vec))

    def add_convex_term(self, conv_fun):
        """Add the convex term `conv_fun` to the problem."""
        for var in conv_fun.variables:
            vector = self._create_variable_vector(var, var.name)
            self.variables.append(var)
            print("Add var", var.name)
            self.vectors_dict.update({var.name: vector})
            self.vectors.append(vector)
        print("Variables", [v.name for v in self.variables])
        print("Vectors", [v for v in self.vectors])
        objective = conv_fun._apply_objective(self)
        conv_fun._apply_linear_constraints(self)
        print("Applying conic constraints")
        conv_fun._apply_conic_constraints(self)
        # print(objective)
        # conv_fun._apply_on_problem(self)
        self.objectives.append(objective)

    def _set_task_parameters(self):
        assert all(
            [p in self._default_parameters().keys() for p in self.parameters.keys()]
        ), "Available parameters are:\n{}".format(self._default_parameters())

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
        int_info = ["intpnt_iter", "opt_numcon", "opt_numvar", "ana_pro_num_var"]
        double_info = [
            "optimizer_time",
            "presolve_eli_time",
            "presolve_lindep_time",
            "presolve_time",
            "intpnt_time",
            "intpnt_order_time",
            "sol_itr_primal_obj",
            "sol_itr_dual_obj",
        ]
        int_value = [self.task.getintinf(getattr(mosek.iinfitem, k)) for k in int_info]
        double_value = [
            self.task.getdouinf(getattr(mosek.dinfitem, k)) for k in double_info
        ]
        info = dict(zip(int_info + double_info, int_value + double_value))
        info.update(
            {
                "solution_status": str(self.task.getsolsta(mosek.soltype.itr)).split(
                    "."
                )[1]
            }
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
