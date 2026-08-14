"""This script contains functions to perform the feasibility check for a given integer solution. The feasibility check
aims to find flows of workers between tours such that each tour can be executed as scheduled.
"""

from gurobipy import Model, GRB, LinExpr
from copy import deepcopy
import time
from config.config import *

def is_sol_actually_feasible(node_sol, node_tours, workers):
    """Perform feasibility check for given solution.

    Parameters
    ----------
    node_sol: dict
        Maps tour indices to their lambda value
    node_tours: list
        Corresponding list of GH_tour objects
    workers: dict
        Maps the skill levels to the number of available workers with the respective skill level in the given instance
        (without downgrading).

    Returns
    -------
    is_feasible: bool
        Indicates if the solution passes the feasibility check.
    feas_check_time: float
        Runtime of the feasibility check (in seconds)
    setup_time: float
        Runtime of the model setup step (in seconds)
    """

    # Checks if a feasible flow of workers implementing the tours exists
    tours = []
    for i in node_sol:  # get all selected tours
        if node_sol[i] > 1 - eps_gap:
            tours.append(node_tours[i])
    is_feasible, feas_check_time, setup_time = perform_feasibility_check_extended(tours, workers)[1:]

    return is_feasible, feas_check_time, setup_time

def is_sol_actually_feasible_extended(node_sol, node_tours, workers, forbidden_tours, solve_as_dmp):
    """Performs extended feasibility check for given solution. If the solution fails the feasibility check, it uses
    the information provided by the feasibility check about the usage of slack workers to compute skill compositions
    for each tour in the current solution. The goal of this step is to provide skill compositions for all current solution
    tours such that the solution is 'almost' disaggregated-feasible, i.e. it exceeds the available disaggregated
    workforce by as little as possible. This is intended to increase the speed at which disaggregated-feasible solutions
    are found by the (subsequently solved) DMP.

    Parameters
    ----------
    node_sol: dict
        Maps tour indices to their lambda value
    node_tours: list
        Corresponding list of GH_tour objects
    workers: dict
        Maps the skill levels to the number of available workers with the respective skill level in the given instance
        (without downgrading).
    forbidden_tours: list
        List of GH_tours forbidden at current branching tree node
    solve_as_dmp: bool
        Indicates if the current node is solved as a DMP (True) or an AMP (False).

    Returns
    -------
    node_tours: list
        List of input node_tours with adjusted skill compositions
    changed_tours_hashes_idxs: list
        List of indices of tours in node_tours whose skill composition (and thus hash) has changed
    changed_tours_hashes: list
        List of corresponding changed hashes
    is_feasible: bool
        Indicates if the solution passes the feasibility check.
    feas_check_time: float
        Runtime of the feasibility check (in seconds)
    setup_time: float
        Runtime of the model setup step (in seconds)
    """

    # 1. Check if a feasible flow of workers implementing the tours exists
    tours = []
    idx_map = {}
    idx = 0
    for i in node_sol:  # get all selected tours
        if node_sol[i] > 1 - eps_gap:
            tours.append(node_tours[i])
            idx_map[idx] = i
            idx += 1
    # solve extended feasibility problem
    x, is_feasible, feas_check_time, setup_time = perform_feasibility_check_extended(tours, workers)

    # 2. get skill comps. of tours from feasibility check and adjust current tour's skill comps.
    changed_tours_hashes = []
    changed_tours_hashes_idxs = []
    # 2.1 get formations without downgrading for each integer tour
    formations = {}
    for i in idx_map:
        formations[i] = {}
        formation_i = node_tours[idx_map[i]].formation_id.lstrip("f_").split(",")
        for s in formation_i:
            formations[i][int(s.split(":")[0])] = int(s.split(":")[1])

    # 2.2 reset skill compositions
    new_skill_comps = {}
    new_skill_comps_cnts = {}
    for i in range(len(tours)):
        tour = tours[i]
        if not tour.is_fake_tour:
            new_skill_comps[i] = {}
            new_skill_comps_cnts[i] = {}
            for k in tour.skill_comp:
                new_skill_comps[i][k] = {}
                new_skill_comps_cnts[i][k] = 0
                for kk in tour.skill_comp[k]:
                    new_skill_comps[i][k][kk] = 0
        else:
            new_skill_comps[i] = "s_fake"
            new_skill_comps_cnts[i] = "s_fake"

    # 2.3 sort x w.r.t. skill level
    x = sorted(x.items(), key = lambda x: x[0][2])

    # 2.4 get new skill compositions and skill composition counts based on x values
    for ((i, j, k), x_val) in x:
        # Skip computation for fake tour: if fake tour is part of the solution, instance will terminate as infeasible anyway.
        # If it is not, we do not care about its skill composition because it's not part of the (possibly optimal) solution
        if tours[j].is_fake_tour:
            continue
        lower_skill_levels = [kk for kk in formations[j] if kk <= k]
        workers_left_to_distribute = x_val
        for kk in lower_skill_levels:   # downgrade as little as possible
            delta = formations[j][kk] - sum(new_skill_comps[j][kk].values())
            if delta > 0:
                new_skill_comps[j][kk][k] += min(delta, workers_left_to_distribute)
                new_skill_comps_cnts[j][k] += min(delta, workers_left_to_distribute)
                workers_left_to_distribute -= min(delta, workers_left_to_distribute)

    # 2.5 For tours that contain and active tour from prior segment: skill composition remains unchanged
    # Sanity-check against computed skill comp cnt (skill comp might change, but that doesn't matter)
    for i in idx_map:
        if node_tours[idx_map[i]].skill_comp_frozen:
            assert new_skill_comps_cnts[i] == node_tours[idx_map[i]].skill_comp_cnt
            new_skill_comps[i] = node_tours[idx_map[i]].skill_comp
            new_skill_comps_cnts[i] = node_tours[idx_map[i]].skill_comp_cnt


    # 2.6 filter out any tours that could be converted to forbidden or forced tours if their skill compositions are changed
    for tour in forbidden_tours:
        indexes_to_remove = []
        forb_tour_hash = tour.get_hash_no_comp()
        for i in idx_map:
            if node_tours[idx_map[i]].get_hash_no_comp() == forb_tour_hash:
                indexes_to_remove.append(i)
        for i in indexes_to_remove:
            del idx_map[i]

    # 2.7 map new skill comps. and skill comp cnts. to node_tours
    # only done when current node is not a DMP node, as DMP nodes already have suitable skill comps.
    if not solve_as_dmp:
        cnt = 0
        for i in idx_map:
            node_tours[idx_map[i]].skill_comp = new_skill_comps[i].copy()
            node_tours[idx_map[i]].skill_comp_cnt = new_skill_comps_cnts[i].copy()
            changed_tours_hashes.append(node_tours[idx_map[i]].get_hash_no_comp())
            changed_tours_hashes_idxs.append(idx_map[i])
            cnt += 1
        print(f"changed skill compositions for {cnt} tours.")

    return node_tours, changed_tours_hashes_idxs, changed_tours_hashes, is_feasible, feas_check_time, setup_time


def perform_feasibility_check_extended(tours, workers_in):
    """Performs a feasibility check that verifies if a solution is disaggregated-feasible, i.e., feasible for the DMP.
     proposed by Dall'Olio & Kolisch (2023).
    Basic principles are identical to Dall'Olio & Kolisch (2023). This implementation introduces additional slack
    workers, which ensure that there always exists a feasible solution. The goal is to provide a flow of workers
    through the network that ensures that each task can be executed as planned. The objective is to minimize the
    number of slack workers, i.e., workers exceeding the available workforce.
    It holds that a solution to the AMP passes this feasibility check iff. the objective value of the following problem
    is 0.

    Parameters
    ----------
    tours: list
        List of GH_tour objects that comprise a given feasible integer solution to the AMP.
    workers_in: dict
        Maps the skill levels to the number of available workers with the respective skill level in the given instance
        (without downgrading).

    Returns
    -------
    x_total: dict
        Maps worker flow decision variables to their optimal solution values. Only contains nonzero variables.
    is_feasible: bool
        Indicates if the solution passes the feasibility check.
    feas_check_time: float
        Runtime of the feasibility check (in seconds)
    setup_time: float
        Runtime of the model setup step (in seconds)
    """

    start_setup = time.time()
    workers = deepcopy(workers_in)

    # 1. create tour data and variables
    i_0 = "i0"      # origin o
    i_np1 = "i_np1" # destination \bar{o}

    # 1.1 store tour indices in list
    J = []      # list of tours
    J0N = []    # list of tours + origin i_0 and destination i_np1
    for r in range(len(tours)):
        J.append(str(r))
        J0N.append(str(r))
    J0N.append(i_0)
    J0N.append(i_np1)
    K = []      # list of skill levels
    for k in workers.keys():
        K.append(k)

    # 1.2 get worker demand of each tour
    demand_w_d = {}     # keys: indices of tours, values: dicts with worker demand per skill level (with downgrading)
    demand = {}         # keys: indices of tours, values: dicts with worker demand per skill level (without downgrading), only needed for active tours
    for r in J:
        demand_w_d[r] = {}
        demand[r] = {}
        for k in K:
            demand_w_d[r][k] = 0
            demand[r][k] = 0
        for k in tours[int(r)].formation_w_d:
            demand_w_d[r][k] = tours[int(r)].formation_w_d[k]
        for k in tours[int(r)].skill_comp_cnt:
            demand[r][k] = tours[int(r)].skill_comp_cnt[k]

    # 1.3 get successors and predecessors of each tour
    S = {}      # keys: indices of tours, values: list of successor tours
    P = {}  # keys: indices of tours, values: list of predecessor tours
    for r in J:
        S[r] = set()
        P[r] = set()
    S[i_0] = set()
    P[i_np1] = set()
    S[i_0].add(i_np1)
    P[i_np1].add(i_0)
    # set each tour as successor/predecessor of origin/destination and get all successor/predecessors among other tours
    for r1 in J:
        S[r1].add(i_np1)
        P[r1].add(i_0)
        S[i_0].add(r1)
        P[i_np1].add(r1)
        for r2 in J:
            if tours[int(r1)].quantile_return_time <= tours[int(r2)].leave_time:
                S[r1].add(r2)
                P[r2].add(r1)

    model = Model('flow_MIP')
    model.setParam("LogToConsole", False)

    # 1.4 for each tuple (tour, tour, skill level): create non-negative integer variable
    x = model.addVars(J0N, J0N, K, lb=0.0, vtype=GRB.INTEGER, name="x")
    x_slack = model.addVars(J0N, J0N, K, lb=0.0, vtype=GRB.INTEGER, name="slack_")

    # 2. set model constraints
    demand_met_constraint = {}  # worker demand met
    skill_comp_demand_met_constraint = {}  # worker demand met for tours with frozen skill comps. (tours containing active tours from prior segments)
    flow_conservation_constraint = {}   # outflow of workers = inflow of workers per skill level
    flow_conservation_constraint_slack = {}   # outflow of workers = inflow of workers per skill level (only counting external workers)
    source_constraint = {}      # all workers leave (and return to) the depot

    # 2.1 workforce demand constraints
    for j in J:
        for k in K:
            # for tours with frozen skill compositions: need to enforce the exact skill composition
            if tours[int(j)].skill_comp_frozen:
                demand_met_cons = LinExpr()
                for i in P[j]:
                    demand_met_cons += x[(i, j, k)] + x_slack[(i, j, k)]
                skill_comp_demand_met_constraint[(j, k)] = model.addLConstr(demand_met_cons == demand[j][k],
                                                                            f"skill_comp_demand_met_tour{j}_level{k}")
            # for all other tours: add demand constraint with downgrading instead
            else:
                demand_met_cons = LinExpr()
                for i in P[j]:
                    for kk in list(filter(lambda x: x >= k, K)):
                        demand_met_cons += x[(i, j, kk)] + x_slack[(i, j, kk)]
                demand_met_constraint[(j, k)] = model.addLConstr(demand_met_cons >= demand_w_d[j][k],
                                                                 f'demand_met_tour{j}_level{k}')

    # 2.2 worker flow conservation
    for i in J:
        for k in K:
            flow_conservation_cons = LinExpr()
            flow_conservation_cons_slack = LinExpr()
            for h in P[i]:
                flow_conservation_cons += x[(h, i, k)]
                flow_conservation_cons_slack += x_slack[(h, i, k)]
            for j in S[i]:
                flow_conservation_cons -= x[(i, j, k)]
                flow_conservation_cons_slack -= x_slack[(i, j, k)]
            flow_conservation_constraint[(i, k)] = model.addLConstr(flow_conservation_cons == 0,
                                                                    'flow_conservation')
            flow_conservation_constraint_slack[(i, k)] = model.addLConstr(flow_conservation_cons_slack == 0,
                                                                    'flow_conservation_slack')

    # 2.3 Create one source workforce constraint for each time instant: the no. of workers of level k leaving the depot up
    # until time t must not exceed workers[k][t]
    instants = list(workers_in[min(workers_in)].keys())
    for k in K:
        for t in instants:
            source_cons = LinExpr()
            for j in S[i_0]:
                if j == i_np1: # sink node treated separately: always counts to the workforce constraint
                    source_cons += x[(i_0, j, k)]
                else: # else: check if tour already left at or before t
                    if tours[int(j)].leave_time <= t:
                        source_cons += x[(i_0, j, k)]

            source_constraint[(k, t)] = model.addLConstr(source_cons <= workers[k][t], "source_cons_" + str(k))


    # 3. set objective function
    obj = LinExpr()
    for k in K:
        for j in S[i_0]:
            obj += x_slack[(i_0, j, k)] * (1 + 1/(1000-2**k))        # slight perturbation to prefer solutions with workers of smaller skill levels
    model.setObjective(obj, GRB.MINIMIZE)

    # 4. solve model and check if solution passes the feasibility check
    setup_time = time.time() - start_setup
    start_feas_check = time.time()
    model.optimize()
    feas_check_time = time.time() - start_feas_check
    obj_val = model.getAttr(GRB.attr.ObjVal)
    is_feasible = obj_val < eps_gap

    no_additional_workers = 0
    for j in S[i_0]:
        for k in K:
            if x_slack[(i_0, j, k)].X > 1 - eps_global:
                no_additional_workers += x_slack[(i_0, j, k)].X

    # 5.1 get all nontrivial x/x_slack variables not entering the sink
    x_total = {}
    for (i,j,k) in x:
        # 5.1.1 skip edges that enter the source or leave the sink: such arcs are created by the model, but never used
        if i == "i_np1" or j == "i0":
            continue
        # 5.1.2 also skip edges that enter the sink
        if j != "i_np1":
            if x[(i,j,k)].X > eps_global or x_slack[(i,j,k)].X > eps_global:
                if i == "i0":
                    x_total[(i,int(j),k)] = int(round(x[(i,j,k)].X + x_slack[(i,j,k)].X,0))
                else:
                    x_total[(int(i),int(j),k)] = int(round(x[(i,j,k)].X + x_slack[(i,j,k)].X,0))
    return x_total, is_feasible, feas_check_time, setup_time