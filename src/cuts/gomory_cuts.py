"""This file contains functions to compute, update, and extend rank-1 Chvátal Gomory cuts.
"""

import math
import time
from gurobipy import Model, GRB, LinExpr
from config.config import *


def add_gomory_cut_to_existing_node(node, u_task, u_kt, alpha_zero):
    """Add a newly created gomory cut to each other unexplored node. Used when cuts are added to a node and cuts should
    also be applied to all unexplored nodes in the branching tree.
    Note: this function is not used by Hagn et al. (2026), because cuts are only added at the root node. Computational
    experiments showed that adding cuts ONLY add the root node is computationally more efficient than also adding them
    at subsequent nodes.

    Parameters
    ----------
    node: GH_node
        Target node in the branching tree
    u_task: dict
        Maps tasks to the gomory cut coefficient of the corresponding task execution constraint
    u_kt: dict
        Maps tuples of skill levels and time instants to the gomory cut coefficient of the corresponding workforce
        constraint
    alpha_zero: float
        Right hand-side of the gomory cut
    """

    # 1. update coefficients per constraint lists in node
    node.u_task.append(u_task)
    node.u_kt.append(u_kt)

    # 2. compute coefficient for each tour and store
    gomory_cuts_lhs = {}
    for tour_idx in range(len(node.tours)):
        tour = node.tours[tour_idx]
        alpha_tour = sum([u_task[task] for task in tour.tasks])
        entries_u_kt = [(k,t) for (k,t) in u_kt if t >= tour.leave_time and t < tour.quantile_return_time and k in tour.formation_w_d]
        alpha_tour += sum([u_kt[(k, t)] * tour.formation_w_d[k] for (k,t) in entries_u_kt])
        alpha_tour = math.floor(alpha_tour)
        if abs(alpha_tour) > eps_global:
            gomory_cuts_lhs[tour_idx] = alpha_tour
    node.gomory_cuts_lhs.append(gomory_cuts_lhs)
    node.gomory_cuts_rhs.append(alpha_zero)


def extend_gomory_cuts(neg_tours, gomory_cuts_lhs, u_task, u_kt, tours):
    """Extend existing gomory cuts onto new tours by calculating their coefficient. Used when a new tour is added to a
    node that has at least one gomory cut.

    Parameters
    ----------
    neg_tours: list
        List of newly added GH_tour tours with negative reduced cost
    gomory_cuts_lhs: dict
        Maps the indices of gomory cuts to a list that contains the left hand-side coefficients of all current tours
        for the given gomory cut
    u_task: dict
        Maps tasks to the gomory cut coefficient of the corresponding task execution constraint
    u_kt: dict
        Maps tuples of skill levels and time instants to the gomory cut coefficient of the corresponding workforce
        constraint
    tours: list
        List of GH_tour tours at the current node

    Returns
    -------
    gomory_cuts_lhs: dict
        Maps the indices of gomory cuts to a list that contains the left hand-side coefficients of all current tours
        for the given gomory cut, enriched by coefficient info for the new tours 'neg_tours'
    """

    for i in range(len(gomory_cuts_lhs)):
        cut_u_task = u_task[i]
        cut_u_kt = u_kt[i]
        for tour_idx in range(len(neg_tours)):
            tour = neg_tours[tour_idx]
            alpha_tour = sum([cut_u_task[task] for task in tour.tasks])
            entries_u_kt = [(k,t) for (k,t) in cut_u_kt if t >= tour.leave_time and t < tour.quantile_return_time and k in tour.formation_w_d]
            alpha_tour += sum([cut_u_kt[(k, t)] * tour.formation_w_d[k] for (k,t) in entries_u_kt])
            alpha_tour = math.floor(alpha_tour)
            if abs(alpha_tour) > eps_global:
                gomory_cuts_lhs[i][len(tours)+tour_idx] = alpha_tour
    return gomory_cuts_lhs



def check_for_gomory_cuts_only_nonzero_light(node, opt_sol, workers_kt):
    """Find gomory cuts as described by Fischetti & Lodi (2006).
    This function only considers selected tours (i.e. lambda^r > 0). All other tours do not have a contribution to
    gomory cuts at the current node, so we can ignore them.
    Also we do not include branching constraints or existing gomory cuts into our formulation, as they are not
    necessary to find valid gomory cuts and in fact weaken the gomory cut strength.

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    opt_sol: dict
        Maps tour indices to their optimal lambda value
    workers_kt: dict
        Maps tuples of skill levels and time instants to the number of available workers. If solve_as_dmp is
        True, this worker count is equal to the worker without downgrading, otherwise it equals the worker count
        with downgrading.
    selected_tours_idxs: list
        List of indices of tours in node.tours that have a nonzero lambda value

    Returns
    -------
    found_violated_cut: bool
        True iff. a violated gomory cut was found
    opt_alpha: float | None
        Left hand-side of the cut at the current solution (if violated cut was found), else None
    opt_alpha_zero: float | None
        Right hand-side of cut if a violated cut was found, else None
    opt_u_task: dict | None
        Maps tasks to the gomory cut coefficient of the corresponding task execution constraint if a violated cut was
         found, else None
    opt_u_kt: dict | None
        Maps tuples of skill levels and time instants to the gomory cut coefficient of the corresponding workforce
        constraint if a violated cut was found, else None

    """
    # 1. get indices of tours that were selected in current solution
    selected_tours_idxs = [i for i in opt_sol if opt_sol[i] > eps_global]

    # 2. initialize MIP to find cuts
    model = Model("Gomory Cuts")
    model.setParam("LogToConsole", False)
    model.setParam("Method", 1)
    model.setParam("Threads", 1)
    model.setParam("ConcurrentMIP", 1)
    model.setParam("WorkLimit", .3)
    # model.setParam("TimeLimit", .3)
    model.setParam("Seed", 42069)

    w_pertub = eps_global / 100 # slightly perturb objective function to improve MIP performance
    precision = eps_gc_recalc       # set high precision to avoid rounding errors leading to infeasible cuts

    # 3. add variables
    # 3.1 integer variables for each tour and one constant
    alpha = model.addVars(selected_tours_idxs, vtype = GRB.INTEGER, name = "alpha")
    alpha_zero = model.addVar(vtype = GRB.INTEGER, name = "alpha_zero")
    # 3.2 continuous variables for each constraint (=coefficients in gomory cut)
    u_task = model.addVars(node.inst.tasks, lb = 0., ub = 1 - precision, vtype = GRB.CONTINUOUS, name = "u_task")
    u_kt = {}
    for k in workers_kt:
        for t in workers_kt[k]:
            u_kt[(k,t)] = model.addVar(lb = 0., ub = 1 - precision, vtype = GRB.CONTINUOUS, name = f"u_{k},{t}")
    # 4. add constraints
    # 4.1 constraints (7) from Fischetti & Lodi (2006) for each variable
    var_constraints_leq = {}
    var_constraints_geq = {}
    for tour_idx in selected_tours_idxs:
        tour = node.tours[tour_idx]
        var_cons = LinExpr()
        # workforce part of constraints
        for k in tour.formation_w_d:
            for t in range(tour.leave_time, tour.quantile_return_time):
                var_cons += tour.formation_w_d[k] * u_kt[(k,t)]
        # task part of constraints
        for task in tour.tasks:
            var_cons += u_task[task]
        var_cons -= alpha[tour_idx]
        var_constraints_leq[tour_idx] = model.addConstr(var_cons <= 1 - precision, f"variable_cons_{tour_idx}")
        var_constraints_geq[tour_idx] = model.addConstr(var_cons >= 0, f"variable_cons_{tour_idx}")

    # 4.2 constraints (8) from Fischetti & Lodi (2006) for constant alpha_zero
    alpha_zero_cons = LinExpr()
    for (k, t) in u_kt:
        alpha_zero_cons += u_kt[(k,t)] * workers_kt[k][t]
    for task in u_task:
        alpha_zero_cons += u_task[task]
    alpha_zero_cons -= alpha_zero
    model.addConstr(alpha_zero_cons <= 1 - precision , "alpha_zero_cons")
    model.addConstr(alpha_zero_cons >= 0, "alpha_zero_cons")

    # 5. set objective
    obj = LinExpr()
    for tour_idx in selected_tours_idxs:
        obj += alpha[tour_idx] * opt_sol[tour_idx]
    # slightly perturbate the objective to ensure results consistency across runs
    for i, task in enumerate(sorted(u_task)):  # sorted = deterministic ordering
        obj -= (w_pertub * (1 + i * 1e-6)) * u_task[task]

    for i, kt in enumerate(sorted(u_kt)):
        obj -= (w_pertub * (1 + i * 1e-6)) * u_kt[kt]
    obj -= alpha_zero
    model.setObjective(obj, GRB.MAXIMIZE)

    # 6. optimize model and parse output
    start_time = time.time()
    model.optimize()
    runtime_gomory = time.time() - start_time

    # 7. if model has been solved to optimality: check if constraint is violated by current solution and if so,
    # return coefficients
    found_violated_cut = False
    opt_alpha = None
    opt_alpha_zero = None
    opt_u_task = None
    opt_u_kt = None
    if model.getAttr(GRB.attr.SolCount) > 0:
        opt_val = model.getAttr(GRB.attr.ObjVal)
        if opt_val > 0:
            found_violated_cut = True
            opt_alpha = model.getAttr(GRB.attr.X, alpha)
            for key in opt_alpha:
                opt_alpha[key] = round(opt_alpha[key], 0)
            opt_alpha_zero = round(alpha_zero.X,0)
            opt_u_task = model.getAttr(GRB.attr.X, u_task)
            opt_u_kt = model.getAttr(GRB.attr.X, u_kt)
            # only return nonzero u_kt values
            u_kt_nonzero = {}
            for (k,t) in opt_u_kt:
                if opt_u_kt[(k,t)] > eps_global:
                    u_kt_nonzero[(k,t)] = opt_u_kt[(k,t)]
            opt_u_kt = u_kt_nonzero


    print(f"Runtime GC-SEP: {runtime_gomory}, objective value {opt_val}")

    return found_violated_cut, opt_alpha, opt_alpha_zero, opt_u_task, opt_u_kt
