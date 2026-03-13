"""This script contains the class Master_model, which defines the master problem solved at every pricing iteration at
every node in the branching tree.
It also contains the function solve_heuristic_master, which is called when the BPC&S runtime limit is reached and
a heuristic solution is seeked.
"""

import math
import time
import copy
from gurobipy import Model, GRB, LinExpr, Column
from src.pricing.utils import get_all_skill_comps
from src.core.utils import get_setup_pricing
from config.config import *

class Master_model():
    def __init__(self, tasks, tours, indices, workers_kt, t_gr, t_le, forced_arcs, infeas_aggr_sol_sets, var_type,
                 solve_as_dmp, gomory_cuts_lhs, gomory_cuts_rhs, cores_per_thread):
        """
        Initialize the master problem used throughout the algorithm as a gurobi model. This model is set up anew
        whenever a new node explored.

        Parameters
        ----------
        tasks: list
            List of all tasks of the problem instance.
        tours: list of GH_tour objects
            List of all tours present at the current node in the branching tree.
        indices: list
            List of indices of all tours present at the current node in the branching tree.
        workers_kt: dict
            Maps tuples of skill levels and time instants to the number of available workers. If solve_as_dmp is
            True, this worker count is equal to the worker without downgrading, otherwise it equals the worker count
            with downgrading.
        t_gr: dict
            Maps time instants to the minimum number of active tours (yielded by branching on tour counts).
        t_le: dict
            Similar to t_gr, but corresponds to <= constraints (i.e., maximum number of active tours at a time instant).
        forced_arcs: list
            List of arc tuples (from, to, leave_time) that must be used (only active if approach by Yuan et al. (2015) is used).
        infeas_aggr_sol_sets: list
            List of disaggregated-infeasible solutions (only active if DMP is not used, i.e. use_dmp = False).
        var_type: grb.GRB.BINARY or grb.GRB.CONTINUOUS
            Variable type to be used for decision variables lambda.
        solve_as_dmp: bool
            Indicates if the current node is solved as a DMP (True) or an AMP (False).
        gomory_cuts_lhs: dict
            Maps gomory cut indices to their left hand side expressions (formatted as grb.constraints).
        gomory_cuts_lhs: dict
            Contains the corresponding right hand sides.
        cores_per_thread: int
            Number of cores to use by gurobi when solving the master problem.

        Note: if tours are forced, workers_kt already takes their workforce requirements into account and is thus
        reduced by the respective workforce requirements (as these workers are no longer available at certain time instants)
        """

        self.master_setup_time = 0

        # 0. initialize model
        master_setup_start = time.time()
        self.model = Model("master")
        self.model.setParam("LogToConsole", False)
        if cores_per_thread is not None:
            self.model.setParam("Threads", cores_per_thread)
        self.model.setParam("Method", 4)
        self.model.setParam("ConcurrentMIP", 1)

        # 1. create variables and constraint
        self.lbda = self.model.addVars(indices, lb=0.0, vtype=var_type, name="lbda")
        self.exec_constraints = {}      # task execution constraints
        self.workers_constraints = {}   # workforce constraints
        self.t_gr_constraints = {}      # constraints for tour count at given timestep
        self.t_le_constraints = {}
        self.arcs_gr_constraints = {}   # constraints for forced arcs (sum == 1)
        self.arcs_le_constraints = {}
        self.infeas_aggr_sol_constraints = {}   # constraints from solution forbidden by the feasibility check
        self.gomory_cuts_constraints = {}       # gomory cuts

        # 2. add constraints
        # 2.1 execution constraints
        for task in tasks:
            exec_cons = LinExpr()
            for i in indices:
                if task in tours[i].tasks:
                    exec_cons += self.lbda[i]
            self.exec_constraints[task] = self.model.addConstr(exec_cons >= 1, "exec_task_" + str(task))
            
        # 2.2 worker constraints
        # if solve_as_dmp is set to true: consider workers on a disaggregated level
        if solve_as_dmp:
            for k in workers_kt:
                for t in workers_kt[k]:
                    workers_cons = LinExpr()
                    for i in indices:  # indices = list of tour indices (0,...,len(tours)-1)
                        if t in list(range(tours[i].leave_time, tours[i].quantile_return_time)) and k in tours[i].skill_comp_cnt:
                            workers_cons -= tours[i].skill_comp_cnt[k] * self.lbda[i]
                    self.workers_constraints[(k, t)] = self.model.addConstr(workers_cons >= -workers_kt[k][t],
                                                                            "level_" + str(k) + "_instant_" + str(t))

        else:
            for k in workers_kt:
                for t in workers_kt[k]:
                    workers_cons = LinExpr()
                    for i in indices:   # indices = list of tour indices (0,...,len(tours)-1)
                        if t in list(range(tours[i].leave_time, tours[i].quantile_return_time)) and k in tours[i].formation_w_d:
                            workers_cons -= tours[i].formation_w_d[k]*self.lbda[i]
                    self.workers_constraints[(k,t)] = self.model.addConstr(workers_cons >= -workers_kt[k][t] ,
                                                                           "level_"+ str(k) + "_instant_" + str(t))
        
        # 2.3 t_gr constraints (from branching on no. of tours)
        for t in t_gr:
            t_gr_cons = LinExpr()
            for i in indices:
                if t in list(range(tours[i].leave_time, tours[i].quantile_return_time)):
                    # 2.3.1 fake tour is treated differently: counts as t_gr[t] many tours to ensure feasibility
                    if tours[i].is_fake_tour:
                        t_gr_cons += t_gr[t] * self.lbda[i]
                    else:
                        t_gr_cons += self.lbda[i]
            self.t_gr_constraints[t] = self.model.addConstr(t_gr_cons >= t_gr[t],
                                                            "branching_constraint:t=" + str(t) + ">=" + str(t_gr[t]))
    
        # 2.4 t_le constraints (from branching on no. of tours)
        for t in t_le:
            t_le_cons = LinExpr()
            for i in indices:
                if t in list(range(tours[i].leave_time, tours[i].quantile_return_time)):
                    # 2.4.1 fake tour is treated differently: counts as t_gr[t] many tours to ensure feasibility
                    if tours[i].is_fake_tour:
                        t_le_cons -= t_le[t] * self.lbda[i]
                    else:
                        t_le_cons -= self.lbda[i]
            self.t_le_constraints[t] = self.model.addConstr(t_le_cons >= t_le[t]*(-1),
                                                            "branching constraint:t=" + str(t) + "<=" + str(t_le[t]))

        # 2.5 forced arcs constraints
        for arc in forced_arcs:
            forced_arc_cons = LinExpr()
            for i in indices:
                if i == 0:
                    forced_arc_cons += self.lbda[0]  # always add the fake tour to the constraint to ensure feasibility
                    continue
                tour = tours[i]
                sequence = ["source"] + tour.tasks + ["sink"]
                for it in range(len(sequence)-1):
                    if (sequence[it], sequence[it+1], tour.quantile_finish_time[sequence[it+1]]) == arc:
                        forced_arc_cons += self.lbda[i]
            self.arcs_gr_constraints[arc] = self.model.addConstr(forced_arc_cons >= 1, f"forced arc {arc} >= 1")
            self.arcs_le_constraints[arc] = self.model.addConstr(-forced_arc_cons >= -1, f"forced arc {arc} <= 1")


        # 2.6 add no-good cuts (from failed feasibility checks)
        for s in range(len(infeas_aggr_sol_sets)):
            infeas_aggr_sol_cons = LinExpr()
            for i in infeas_aggr_sol_sets[s]:
                infeas_aggr_sol_cons += self.lbda[i]
            self.infeas_aggr_sol_constraints[s] = self.model.addConstr(infeas_aggr_sol_cons <= len(infeas_aggr_sol_sets[s])-1,
                                                                       "infeasible_aggr_sol_set " + str(s))


        # 2.7 add gomory cuts
        for i in range(len(gomory_cuts_lhs)):
            gomory_cut_cons = LinExpr()
            for tour_idx in gomory_cuts_lhs[i]:
                gomory_cut_cons -= gomory_cuts_lhs[i][tour_idx] * self.lbda[tour_idx]
            self.gomory_cuts_constraints[i] = self.model.addConstr(gomory_cut_cons >= gomory_cuts_rhs[i]*(-1),
                                                                   f"gomory_cut_{i}")
        # 3. add objective function
        obj = LinExpr()
        for i in indices:
            obj += tours[i].cost*self.lbda[i]
        self.model.setObjective(obj, GRB.MINIMIZE)

        self.master_setup_time = time.time() - master_setup_start    
        
    def add_tours(self, tours, solve_as_dmp, gomory_cuts_lhs):
        """Add tours found in column generation step.
        Parameters
        ----------
        tours: list
            List of GH_tour objects found in column generation/pricing step.
        solve_as_dmp: bool
            Indicates if the current node is solved as a DMP (True) or an AMP (False).
        gomory_cuts_lhs: dict
            Maps gomory cut indices to their left hand side expressions (formatted as grb.constraints).
        """

        master_setup_start = time.time()

        # for each tour found: add tour to master problem by adjusting all relevant constraints
        # this avoids significant overhead that would be generated if the master problem would be set up from scratch
        tour_idx = len(self.lbda) - 1
        for tour in tours:
            tour_idx += 1   # index of current tour
            coeff = []
            cons = []
            # get relevant constraints and respective coefficients of current tour
            # 1. task covering constraints
            for task in tour.tasks:
                coeff.append(1.0)
                cons.append(self.exec_constraints[task])
            # 2. workforce constraints
            if solve_as_dmp:
                for t in range(tour.leave_time, tour.quantile_return_time):
                    for k in tour.skill_comp_cnt:
                        coeff.append(tour.skill_comp_cnt[k]*(-1))
                        cons.append(self.workers_constraints[(k,t)])
            else:
                for t in range(tour.leave_time, tour.quantile_return_time):
                    for k in tour.formation_w_d:
                        coeff.append(tour.formation_w_d[k]*(-1))
                        cons.append(self.workers_constraints[(k,t)])
            # 3. tour count
            for t in range(tour.leave_time, tour.quantile_return_time):
                if t in self.t_gr_constraints:
                    coeff.append(1.0)
                    cons.append(self.t_gr_constraints[t])
                if t in self.t_le_constraints:
                    coeff.append(-1.0)
                    cons.append(self.t_le_constraints[t])
            # 4. forced arcs
            sequence = ["source"] + tour.tasks + ["sink"]
            for i in range(len(sequence) -1):
                if (sequence[i], sequence[i+1], tour.quantile_finish_time[sequence[i+1]]) in self.arcs_gr_constraints:
                    coeff.append(1.0)
                    coeff.append(-1.0)
                    cons.append(self.arcs_gr_constraints[(sequence[i], sequence[i+1], tour.quantile_finish_time[sequence[i+1]])])
                    cons.append(self.arcs_le_constraints[(sequence[i], sequence[i+1], tour.quantile_finish_time[sequence[i+1]])])
            # 5. gomory cuts
            for i in range(len(self.gomory_cuts_constraints)):
                appended_constraint = False     # set to True once gomory cut is added to cons to avoid adding constraint multiple times
                if tour_idx in gomory_cuts_lhs[i]:
                    coeff.append(gomory_cuts_lhs[i][tour_idx]*(-1))
                    if not appended_constraint:
                        appended_constraint = True
                        cons.append(self.gomory_cuts_constraints[i])

            # add variable for new tour and contribution to constraints
            self.lbda[len(self.lbda)] = self.model.addVar(obj=tour.cost, column=Column(coeff, cons), name = f"lbda[{tour_idx}]")

        self.model.update()
        self.master_setup_time += time.time() - master_setup_start


    def add_gomory_cut(self, gomory_cut_lhs, gomory_cut_rhs):
        """Add a newly found Gomory cut to the reduced master problem. Is called after column generation has terminated
        and a violated gomory cut has been found.

        Parameters
        ----------
        gomory_cut_lhs: dict
            Maps indices of current tours to their left hand-side of the new gomory cut
        gomory_cut_rhs: float
            Right hand-side of the new cut.
        """
        master_setup_start = time.time()
        # 1. set up constraint
        i = len(self.gomory_cuts_constraints)
        gomory_cut_cons = LinExpr()
        for tour_idx in gomory_cut_lhs:
            gomory_cut_cons -= self.lbda[tour_idx] * gomory_cut_lhs[tour_idx]
        # 2. add constraint to master and update model
        self.gomory_cuts_constraints[i] = self.model.addConstr(gomory_cut_cons >= gomory_cut_rhs*(-1),
                                                               f"gomory_cut_{i}")
        self.model.update()
        self.master_setup_time += time.time() - master_setup_start


    def optimize_master(self, return_duals):
        """Start optimization process for master problem.

        Parameters
        ----------
        return_duals: bool
            Indicates if dual values should be returned

        Returns
        -------
        opt_sol: dict
            Maps tour indices to their optimal lambda value
        opt_val: float
            Optimal objective value
        mu: dict
            Maps task execution constraints to their dual values
        delta: dict
            Maps workforce constraints to their dual values
        rho_gr: dict
            Maps >= tour count constraints to their dual values
        rho_le: dict
            Maps >= tour count constraints to their dual values
        psi: dict
            Maps gomory cuts to their dual values
        zeta_le: dict
            Maps <= 1 arc usage constraints to their dual values
        zeta_gr: dict
            Maps >= 1 arc usage constraints to their dual values
        tot_time: float
            Total solving time (in seconds)

        """

        # 1. optimize and log runtime
        start_time = time.time()
        self.model.optimize()
        tot_time = time.time() - start_time

        # 2. if problem feasible: return optimal solution
        if self.model.getAttr(GRB.attr.SolCount) > 0:
            # 2.1 get objective value and variable values
            opt_sol = self.model.getAttr(GRB.attr.X, self.lbda)
            opt_val = self.model.getAttr(GRB.attr.ObjVal)
            # 2.2 if duals are not needed: only return solution
            if not return_duals:
                return opt_sol, opt_val
            # 2.3 get constraint duals
            mu = self.model.getAttr(GRB.attr.Pi, self.exec_constraints)
            delta = self.model.getAttr(GRB.attr.Pi, self.workers_constraints)
            nonzero_delta = {}
            for (k,t) in delta:
                if delta[(k,t)] > eps_global: # filter nonzero entries for delta for leaner delta structure (makes computations in pricing faster)
                    nonzero_delta[(k,t)] = delta[(k,t)]
            delta = nonzero_delta
            rho_gr = self.model.getAttr(GRB.attr.Pi, self.t_gr_constraints)
            rho_le = self.model.getAttr(GRB.attr.Pi, self.t_le_constraints)
            psi = self.model.getAttr(GRB.attr.Pi, self.gomory_cuts_constraints)
            zeta_le = self.model.getAttr(GRB.attr.Pi, self.arcs_le_constraints)
            zeta_gr = self.model.getAttr(GRB.attr.Pi, self.arcs_gr_constraints)
            return opt_sol, opt_val, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr, tot_time

        # 3. else: return objective = inf and empty dicts
        else:
            if not return_duals:
                return {}, math.inf
            return {}, math.inf, {}, {}, {}, {}, {}, {}, {}, tot_time


def solve_heuristic_master(inst, solution, all_tours, root, pricing_networks, cum_forbidden_tours, nodes_count,
                           tl_heur, start_time, elapsed_time, cores_per_thread):
    """Solve the master problem to integrality, given a set of columns. Called at the end of the algorith if runtime
    limit is exceeded.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    solution: GH_solution
        Solution object containing the best integer solution found alongside some statistics
    all_tours: list
        List of all tours found so far during the BPC&S procedure
    root: GH_node
        Root node ofthe branching tree
    pricing_networks: dict
        Maps formation IDs to their workers.pricing.graph.PricingNetwork object
    cum_forbidden_tours: int
        Total no. of tours that were forbidden at at least one node
    nodes_count: int
        Number of branching tree nodes explored so far
    tl_heur: float
        Time limit for heuristic (in seconds).
    start_time: float
        Start timestamp of the algorithm
    elapsed_time: float
        Total runtime required by the algorithm to find the optimal solution
    cores_per_thread: int
        Number of cores to use by gurobi when solving the master problem

    Returns
    -------
    solution: GH_solution
        Solution object containing the best integer solution found alongside some statistics

    """

    # 1. if no heuristic solution wanted: store runtime and return solution
    if tl_heur == 0:
        solution.cannot_solve_time = elapsed_time
        return solution
    # 2. calculate heuristic solution based on current LP solution
    print("Time is up, a heuristic solution using the current tours will be found\n")

    # 3. remove tours with same properties except skill comp.
    tours_no_dup = []
    all_hashes = []
    for tour_i in all_tours:
        all_hashes.append(tour_i.get_hash_no_comp())
    all_hashes = set(all_hashes)
    for tour_i in all_tours:
        hash_i = tour_i.get_hash_no_comp()
        if hash_i in all_hashes:
            tours_no_dup.append(tour_i)
            all_hashes.remove(hash_i)
    all_tours = copy.deepcopy(tours_no_dup)

    # remove fake tour
    all_tours.pop(0)

    # 4. create column for each tour found and each possible skill comp.
    # 4.1 get all skill comps for each formation id
    skill_comps_per_formation = {}
    skill_comps_cnts_per_formation = {}
    for formation_id in inst.formations:
        skill_comps_per_formation[formation_id], skill_comps_cnts_per_formation[
            formation_id], skill_comps_cnts_ids = get_all_skill_comps(inst, formation_id)
    # 4.2 generate all possible tours
    heur_tours = []
    for tour in all_tours:
        for i in range(len(skill_comps_per_formation[tour.formation_id])):
            new_tour = tour.copy()
            new_tour.skill_comp = skill_comps_per_formation[tour.formation_id][i]
            new_tour.skill_comp_cnt = skill_comps_cnts_per_formation[tour.formation_id][i]
            heur_tours.append(new_tour)

    # 5. setup and solve heuristic master problem to integrality
    workers_kt = {}  # workers without downgrading for each instant t
    for k in root.inst.skill_levels:
        workers_kt[k] = {}
        for t in root.inst.instants:
            workers_kt[k][t] = root.inst.workers[k]
    print("solving heuristic master problem.")
    heur_model = Master_model(inst.tasks, heur_tours, list(range(len(heur_tours))), workers_kt,
                                     {}, {}, [], [], GRB.BINARY, True, [], [], cores_per_thread)
    heur_model.model.setParam("TimeLimit", tl_heur)
    heur_model.model.setParam("MIPGap", eps_gap)
    heur_sol, heur_val = heur_model.optimize_master(return_duals=False)[:2]

    # 6. analyze solution
    # get probabilities of TW violations for each task and tour lengths and runtime
    tour_lengths = {}
    tw_viol_probs = {}
    for idx in heur_sol:
        if heur_sol[idx] > 0.999:
            tour = heur_tours[idx]
            if len(tour.tasks) not in tour_lengths:
                tour_lengths[len(tour.tasks)] = 0
            tour_lengths[len(tour.tasks)] += 1
            for task in tour.tasks:
                if task not in inst.tasks:
                    continue
                if tour.tw_viol_prob[task] > eps_global / 100 and tour.tw_viol_prob[task] < 1 - eps_global / 100:
                    tw_viol_probs[task] = tour.tw_viol_prob[task]
    tw_viol_probs = sorted(tw_viol_probs.items(), key=lambda x: x[1], reverse=True)
    tw_viol_probs = dict(tup for tup in tw_viol_probs)
    solution.tw_viol_prob = tw_viol_probs.copy()
    tour_lengths = sorted(tour_lengths.items(), key=lambda x: x[0])
    solution.tour_lengths = dict(tour_lengths)

    # 7. log some runtime stats
    elapsed_time = time.time() - start_time
    solution.time_setup_pricing = get_setup_pricing(pricing_networks)

    # 8. if heuristic obj. val == best lower bound found: heuristic solution is optimal
    solution.tot_columns = len(all_tours)  # total no. of columns used
    solution.avg_forbidden_tours_per_node = cum_forbidden_tours / nodes_count
    if hasattr(root, 'lb') and math.isclose(heur_val, root.lb, abs_tol=1e-07):
        # store as optimal solution
        solution.add_optimal_solution(heur_sol, heur_val, heur_tours, elapsed_time, solution.root_lb)
    # else: store as heuristic (suboptimal) solution
    elif heur_val < math.inf:
        solution.add_heuristic_solution(heur_sol, heur_val, heur_tours, root.lb, elapsed_time, solution.root_lb)
    else:
        solution.cannot_solve_time = elapsed_time
    return solution
