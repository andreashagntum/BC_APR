"""
Pricing network construction and management for the stochastic ESPPRC pricing subproblem.

This module defines the graph structure over which the labeling algorithm in
dynamic_programming.py operates. A separate PricingNetwork is built once per worker
formation at the start of the algorithm and then reused (with lightweight in-place
updates) at every node of the branch-and-bound tree.

Core class: PricingNetwork
    Represents the directed pricing graph for a single worker formation. Key responsibilities:

    Construction (__init__):
      - Builds a networkx DiGraph with one node per compatible task plus source/sink.
      - Copies bin-dependent stochastic travel-time distributions from the instance and
        precomputes min/max/quantile travel_times for fast feasibility lookups.
      - Prunes arcs that are guaranteed to violate the alpha chance constraint or the
        extended time windows for every possible finish time and time bin.
      - Removes arcs (i, j) where visiting j directly after i is always dominated by
        returning to the depot and starting a fresh tour to j (unnecessary waiting rule).
      - Enumerates all feasible skill compositions for the formation and stores the
        default (no-downgrading) composition.

    Resource setup (set_resources_consumption):
      - Attaches a Resource object to every arc, encoding the tail task, a dominance flag
        (whether the formation is non-dominated for that task), and forbidden-tour tracking
        data used to detect and discard labels that would reproduce a forbidden tour. Resources
        do not encode cost accumulation, because costs depend on the arrival time distribution of
        the individual label.

    Pricing (get_sprc):
      - Entry point called by columngeneration.py for each CG iteration.
      - Delegates to spprc_algorithm in dynamic_programming.py and returns the
        negative-reduced-cost tour(s), label/dominance counts, and runtime statistics.

    Tour construction (build_tour):
      - Converts a sink-reaching Label object into a GH_tour, computing worst-case and
        quantile start/finish times for each task along the path.

    Node management (remove_forced_task / restore_removed_tasks):
      - Temporarily removes forced-task nodes and their incident arcs from the graph
        before solving a node, then restores them afterwards so the network can be
        reused at sibling/ancestor nodes without reconstruction.
"""

import time
import networkx as nx
import bisect
from src.pricing.utils import get_all_skill_comps, find_alpha_quantile_pmf
from src.utils.gh_tour import GH_tour
from src.pricing.dynamic_programming import spprc_algorithm
from config.config import alpha_tol

class PricingNetwork():
    """Pricing network for a given profile. Non time-expanded version."""

    def __init__(self, inst, formation_id):
        """Initialize pricing network object.
        Note: Setting up a PricingNetwork requires non-negligible runtime. Therefore, pricing networks are created
        once at the beginning of the algorithm, and then inplace update at every node in the branching tree.
        Because most PricingNetwork information remains unchanged throughout the algorithm, this saves a significant
        amount of runtime.

        Parameters
        ----------
        inst: instance_loader.Instance
            Contains all necessary instance data read from input files.
        formation_id: str
            String that uniquely defines the formation whose pricing network is supposed to be constructed
        """
        start_setup = time.time()

        self.graph = nx.DiGraph()
        self.time_setup = 0
        self.source = "source"
        self.sink = "sink"
        self.graph.source = self.source
        self.graph.sink = self.sink
        self.nodes_set = []  # set of all nodes in the graph
        self.formation_id = formation_id  # name of considered profile
        self.formation = inst.formations_w_d[self.formation_id].copy()
        self.node_to_task = {}  # keys: node names, values: name of corresponding task
        self.taskinstant_to_node = {}  # keys: tuples of tasks and possible start time, value: name of corresponding node
        self.arcs = {}
        self.arcs[self.source] = []
        self.arcs[self.sink] = []
        self.resources = {}
        self.tasks = []
        self.workers = inst.workers # workers per skill level and time instant w/o downgrading, needed for busy cost calculation
        self.reachability = {}  # keys: nodes, values: all tasks that are reachable from node within their time window
        self.auto_reachability = {}  # keys: nodes i, values: all tasks through which i is re-reachable within its time window, only needed for properly removing/adding forced tasks
        self.poss_starts = {}
        self.task_resources_consd = []       # list of tasks for which resources are to be considered during the labeling algorithm (part of state space relaxation)
        # attributes added for stochastic model
        self.bins = inst.bins
        self.bin_per_instant = inst.bin_per_instant
        self.instant_per_bin = inst.instant_per_bin
        self.travel_times_per_bin = {}
        for time_bin in inst.travel_times_per_bin:
            self.travel_times_per_bin[time_bin] = {}
            for (task1,task2) in inst.travel_times_per_bin[time_bin]:        # copy travel_times, include zero values for impossible travel times (needed for start time distribution calculation)
                if task1 in inst.tasks_per_formation_with_domination[formation_id] and task2 in inst.tasks_per_formation_with_domination[formation_id]:
                    self.travel_times_per_bin[time_bin][(task1,task2)] = {i:0 for i in range(min(inst.travel_times_per_bin[time_bin][(task1,task2)]),
                                                                                          max(inst.travel_times_per_bin[time_bin][(task1,task2)])+1)}
                    for travel_time in inst.travel_times_per_bin[time_bin][(task1, task2)]:
                        self.travel_times_per_bin[time_bin][(task1,task2)][travel_time] = inst.travel_times_per_bin[time_bin][(task1,task2)][travel_time]
        # remove unneeded travel_times
        # self.travel_times = inst.travel_times.copy()
        self.min_travel_times_per_bin = {time_bin: {} for time_bin in self.travel_times_per_bin}  # min. travel_times for each pair of tasks (stored separably since used many times throughout the code)
        self.max_travel_times_per_bin = {time_bin: {} for time_bin in self.travel_times_per_bin}  # # max. travel_times for each pair of tasks
        self.quantile_travel_times_per_bin = {time_bin: {} for time_bin in self.travel_times_per_bin}  # min. travel_times for each pair of tasks (stored separably since used many times throughout the code)
        self.weights = inst.weights
        self.weights[self.sink] = 0     # dummy value
        self.removed_tasks = []  # list of tasks that are removed since they are forbidden via branching
        self.removed_arcs = []  # list of corresponding removed arcs
        self.removed_reachability = {}  #
        self.removed_auto_reachability = {}
        self.task_execution_times = {}
        for task in inst.tasks_per_formation_with_domination[formation_id]:
            self.task_execution_times[task] = inst.modes_with_domination[task][formation_id]
        self.task_execution_times[self.sink] = 0    # dummy values
        self.task_execution_times[self.source] = 0    # dummy values
        self.earliest_starts = inst.earliest_start
        self.earliest_starts[self.source] = inst.begin_horizon    # set earliest start of depot nodes to begin_horizon (dummy value)
        self.earliest_starts[self.sink] = inst.begin_horizon      # same for the sink
        self.earliest_finishes = inst.earliest_finish
        self.latest_finishes = inst.latest_finish.copy()   # latest finishes without incurring a penalty
        self.latest_finishes[self.sink] = inst.end_horizon
        self.latest_finishes_viol = inst.latest_finish_viol
        self.latest_starts = {}
        self.latest_starts_viol = {}
        for task in inst.tasks_per_formation_with_domination[formation_id]:
            self.latest_starts[task] = max(self.earliest_starts[task], self.latest_finishes[task] - self.task_execution_times[task])
            self.latest_starts_viol[task] = self.latest_finishes_viol[task] - self.task_execution_times[task]

        self.begin_horizon = inst.begin_horizon
        self.end_horizon = inst.end_horizon
        self.alpha = inst.service_level
        self.check_dominance_time = 0
        self.calculate_start_distr_time = 0

        # 0. preprocessing
        # 0.1 copy travel_times from inst.depot to tasks to travel_times from self.source/self.sink to tasks for easier value retrieving
        for task in inst.tasks_per_formation_with_domination[formation_id]:
            self.tasks.append(task)
            for time_bin in self.travel_times_per_bin:
                self.travel_times_per_bin[time_bin][(self.source, task)] = {i: 0 for i in range(min(inst.travel_times_per_bin[time_bin][(inst.depot, task)]),
                                                                           max(inst.travel_times_per_bin[time_bin][(inst.depot, task)]) + 1)}
                for travel_time in inst.travel_times_per_bin[time_bin][(inst.depot, task)]:
                    self.travel_times_per_bin[time_bin][(self.source, task)][travel_time] = inst.travel_times_per_bin[time_bin][(inst.depot, task)][travel_time]
                self.travel_times_per_bin[time_bin][(task, self.source)] = self.travel_times_per_bin[time_bin][(self.source, task)]
                self.travel_times_per_bin[time_bin][(task, self.sink)] = self.travel_times_per_bin[time_bin][(task, self.source)]
                self.travel_times_per_bin[time_bin][(self.sink, task)] = self.travel_times_per_bin[time_bin][(self.source, task)]

        # 0.2 get min./max./quantile travel_times between pairs of nodes/source/sink
        # because these values are needed frequently for verifying label feasibility, we once-compute them for
        # constant lookup speed
        for time_bin in self.travel_times_per_bin:
            for (i, j) in self.travel_times_per_bin[time_bin].keys():
                self.min_travel_times_per_bin[time_bin][(i, j)] = min(self.travel_times_per_bin[time_bin][(i, j)])
                self.max_travel_times_per_bin[time_bin][(i, j)] = max(self.travel_times_per_bin[time_bin][(i, j)])
                self.quantile_travel_times_per_bin[time_bin][(i, j)] = find_alpha_quantile_pmf(self.travel_times_per_bin[time_bin][(i, j)], inst.worker_quantile)

        # 0.3 get list of formations that can execute any given task
        self.formation_ids_per_task = {}
        for task in self.tasks:
            self.formation_ids_per_task[task] = []
        for formation in inst.tasks_per_formation_with_domination:
            for task in self.tasks:
                if task in inst.tasks_per_formation_with_domination[formation]:
                    self.formation_ids_per_task[task].append(formation)


        # 1. Create nodes: one node for each task
        self.graph.add_node(self.source)
        self.graph.add_node(self.sink)
        for task in inst.tasks_per_formation_with_domination[formation_id]:  # include tasks for which current profile is dominated
            self.graph.add_node(task)

        # 2. create arcs between all pairs of tasks & arcs to source/sink
        # 2.1 tasks from source and to sink
        for task in inst.tasks_per_formation_with_domination[formation_id]:
            self.graph.add_edge(self.source, task)
            self.graph.add_edge(task, self.sink)

        # 2.2 arcs between tasks
        for u in inst.tasks_per_formation_with_domination[formation_id]:
            for v in inst.tasks_per_formation_with_domination[formation_id]:
                if u != v:
                    self.graph.add_edge(u, v)
                    self.graph.add_edge(v, u)

        # 3. remove edges between tasks i and j if it is impossible to satisfy the service level constraint at task j
        # same holds if earliest finish time + worst-case travel time > latest (violated) start time
        no_edges_removed = 0
        for task_i in inst.tasks_per_formation_with_domination[formation_id]:
            for task_j in inst.tasks_per_formation_with_domination[formation_id]:
                if task_i == task_j:
                    continue
                # 3.1 for each possible finish time at task i: get its bin and calculate convolution with corresponding
                # travel time distribution to task j
                # 3.1.1 for each bin: get earliest finish time at task i
                # (->distribution will dominate all distributions with the same bin and a later finish time)
                covered_bins = [] # list of bins for which the earliest finish time at i (and the corresponding PMF) was already found
                satisfies_chance_constr_per_bin = {}
                for t in range(inst.earliest_finish[task_i], inst.latest_finish_viol[task_i] + 1):
                    if inst.bin_per_instant[t] in covered_bins:
                        continue
                    satisfies_chance_constr_per_bin[(t, inst.bin_per_instant[t])] = True
                    covered_bins.append(inst.bin_per_instant[t])
                # 3.1.2 calculate arrival time distribution for all time bins
                for (finish_time_i, time_bin) in satisfies_chance_constr_per_bin:
                    start_time_cdf_j = {}
                    start_time_pmf_j = {}
                    arrivals_at_tail = {}
                    # 3.1.2.1 get start time distribution at task j
                    for travel_time in self.travel_times_per_bin[time_bin][(task_i, task_j)]:
                        arrivals_at_tail[finish_time_i + travel_time] = self.travel_times_per_bin[time_bin][(task_i, task_j)][travel_time]
                    for arrival_time in range(min(arrivals_at_tail), max(arrivals_at_tail)+1):
                        if arrival_time in arrivals_at_tail:
                            if arrival_time <= self.earliest_starts[task_j]:
                                if self.earliest_starts[task_j] not in start_time_pmf_j:
                                    start_time_pmf_j[self.earliest_starts[task_j]] = 0
                                start_time_pmf_j[self.earliest_starts[task_j]] += arrivals_at_tail[arrival_time]
                            else:
                                start_time_pmf_j[arrival_time] = arrivals_at_tail[arrival_time]
                        else:
                            start_time_pmf_j[arrival_time] = 0
                    for t in start_time_pmf_j:
                        start_time_cdf_j[t] = sum([start_time_pmf_j[t2] for t2 in start_time_pmf_j if t2 <= t])
                    start_time_cdf_j = list(start_time_cdf_j.items())
                    # 3.1.2.2 get quantile and check if quantile > latest_start[task_j]
                    alpha_quantile_idx = bisect.bisect(start_time_cdf_j, self.alpha - alpha_tol, key = lambda x: x[1])
                    if start_time_cdf_j[alpha_quantile_idx][0] > inst.latest_start[task_j]:
                        satisfies_chance_constr_per_bin[(finish_time_i, time_bin)] = False
                    # 3.1.2.3 check if latest_start_viol[task_j] is violated when i is finished as early as possible
                    elif inst.earliest_finish[task_i] + self.max_travel_times_per_bin[time_bin][(task_i, task_j)] > inst.latest_start_viol[task_j]:
                        satisfies_chance_constr_per_bin[(finish_time_i, time_bin)] = False

                # 3.2 if edge violates chance constraint OR extended time window for ALL start times and start time bins: remove edge
                if True not in satisfies_chance_constr_per_bin.values():
                    for time_bin in inst.bins:
                        del self.min_travel_times_per_bin[time_bin][(task_i, task_j)]
                        del self.max_travel_times_per_bin[time_bin][(task_i, task_j)]
                        del self.quantile_travel_times_per_bin[time_bin][(task_i, task_j)]
                        del self.travel_times_per_bin[time_bin][(task_i, task_j)]
                    self.graph.remove_edge(task_i, task_j)
                    no_edges_removed += 1

        print(f"formation {formation_id}: {no_edges_removed} edges removed due to infeasibility w.r.t. chance constraint")

        # 4. remove edges (i,j) if ES_j - LF_i >= d_{i,source}(max) + d_{source,j}(max) holds in each bin
        # In such a case, one might as well return to the depot after visiting i and starting a new tour towards j,
        # potentially occupying workforce for a shorter time window than when visiting i and j sequentially by the
        # same team
        edges_removed = 0
        for task_i in inst.tasks_per_formation_with_domination[formation_id]:
            for task_j in inst.tasks_per_formation_with_domination[formation_id]:
                if task_i == task_j:
                    continue
                if (task_i, task_j) not in self.graph.edges:
                    continue
                all_bins_satisfied = True   # True iff. rule is satisfied for all bins
                for time_bin in inst.bins:
                    # check if edge satisfies rule for current time bin
                    if (inst.earliest_start[task_j] - inst.latest_finish_viol[task_i] <
                            self.max_travel_times_per_bin[time_bin][(task_i, "source")] + self.max_travel_times_per_bin[time_bin][("source", task_j)]):
                        all_bins_satisfied = False
                        break
                # if unnecessary waiting time is guaranteed for all bins: remove direct connection
                if all_bins_satisfied:
                    edges_removed += 1
                    self.graph.remove_edge(task_i, task_j)
                    for time_bin in inst.bins:
                        del self.min_travel_times_per_bin[time_bin][(task_i, task_j)]
                        del self.max_travel_times_per_bin[time_bin][(task_i, task_j)]
                        del self.quantile_travel_times_per_bin[time_bin][(task_i, task_j)]
                        del self.travel_times_per_bin[time_bin][(task_i, task_j)]
        print(f"Edges removed due to guaranteed waiting times: {edges_removed}")


        # 5. compute all possible skill compositions
        self.skill_comps, self.skill_comps_cnt, self.skill_comps_cnts_ids, self.skill_comps_ids = get_all_skill_comps(inst, self.formation_id)
        # 6. compute default skill composition
        skill_comp = {}
        skill_comp_cnt = {}
        formation_id = self.formation_id
        formation_dict = inst.formations[formation_id]
        for k in inst.workers:
            if k in inst.formations[formation_id]:
                skill_comp_cnt[k] = inst.formations[formation_id][k]
            else:
                skill_comp_cnt[k] = 0
            skill_comp[k] = {}
            for kk in inst.workers:
                if kk < k:
                    continue
                if kk == k:
                    if k in formation_dict:
                        skill_comp[k][kk] = formation_dict[k]
                    else:   # skill level not needed in formation
                        skill_comp[k][kk] = 0
                else:
                    skill_comp[k][kk] = 0
        self.default_comp = skill_comp
        self.default_comp_cnt = skill_comp_cnt

        self.time_setup += time.time() - start_setup

    def set_resources_consumption(self, inst, forbidden_tours):
        """Set resources along arcs. Resources are consumed whenever a label is extended along an arc.
        Arc resources contain:
        - tail node
        - binary indicating if the formation of the underlying pricing network is dominated for the arc's tail node
        - dictionary forb_res that maps the indices of all forbidden tour that use the given arc to a tuple consisting
          of the corresponding forbidden tour's depot leave time, quantile finish time, and task sequence (used to
          assess if a certain subpath that is extended along the given arc might yield a forbidden tour in the future)

        Note that, because reduced costs depend on arcs as well as finish time distributions, these are not considered
        a task resource. Instead, their value is always computed on-demand when a label is extended along an arc.

        Parameters
        ----------
        inst: instance_loader.Instance
            Contains all necessary instance data read from input files.
        forbidden_tours: list
            List of tours that are forbidden by branching on tours
        """
        start_setup = time.time()
        self.forb_tours = list(forbidden_tours)

        # set up forbidden resources per tour
        forb_res_per_arc = {arc: {} for arc in self.graph.edges}
        for i in range(len(forbidden_tours)):
            # get all forbidden arcs
            forb_seq = [self.source] + forbidden_tours[i].tasks + [self.sink]
            forb_arcs = [(forb_seq[it], forb_seq[it + 1]) for it in range(len(forb_seq) - 1)]
            # for each arc: store leave time of forbidden tour and quantile finish time at tail node
            for arc in forb_arcs:
                if arc not in forb_res_per_arc:
                    forb_res_per_arc[arc] = {}
                forb_res_per_arc[arc][i] = (forbidden_tours[i].leave_time, forbidden_tours[i].quantile_finish_time[arc[1]])

        # store resources in Resources objects
        for arc in self.graph.edges:
            # 1. check if mode is dominated for given task
            non_dom_form = False
            if arc[1] != self.sink:
                if self.formation_id in list(inst.modes[arc[1]].keys()):
                    non_dom_form = True
            # 2. store resources
            self.resources[(arc[0], arc[1])] = Resource(non_dom_form, forb_res_per_arc[arc])

        self.time_setup += time.time() - start_setup


    def get_sprc(self, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr, only_best_tasks, best_task_cnt, t_max_le, t_max_gr,
                 solve_as_dmp,
                 node, yuan_approach):
        """Solve the ESPPRC for the current pricing graph.

        Parameters
        ----------
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
        t_max_le: dict
            Maps tasks to their latest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
        t_max_gr: dict
            Maps tasks to their earliest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
        only_best_tasks: bool
            Indicates if each pricing network should initially only be solved using the tasks with the most negative
            dual value
        solve_as_dmp: bool
            Indicates if the current node is solved as a DMP (True) or an AMP (False).
        node: GH_node
            Current node in the branching tree
        current_sol_tours: list
            List of GH_tours at the current node that were selected in the current LP relaxation's solution (i.e., whose
             current objective value is > 0)
        yuan_approach: bool
            Indicates if the approach by Yuan et al. (2015) should be used. If set to True, overrides no_gomory_cuts = 0,
            branch_on_task_finish_times = False, and use_dmp = True.

        Returns
        -------
        min_path: list
            List of paths corresponding to the negative columns that were found and return
        min_cost: bool
            Reduced cost of the (reduced cost-wise) minimal label
        min_labels: list
            List of negative reduced cost labels that were found
        count_labels: int
            Number of labels created during the pricing step
        count_dom: int
            Number of labels dominated
        tour_costs: float
            Objective function value of the (reduced cost-wise) minimal label
        sink_labels_network: int
            Number of non-dominated labels at the sink during the pricing step
        no_initial_labels: int
            Number of initial labels created by workers.pricing.dynamic_programming.create_initial_labels()
        """

        # 1. for each forbidden tour: get their skill composition
        # when solving the DMP, we do not discard labels because they are equal to a forbidden tour. instead, when
        # ex-post computing the best label, we skip forbidden skill compositions
        forb_skill_comps = {}       # list of skill comps of each forbidden tour
        forb_tour_idxs = list(range(len(self.forb_tours))) # also get their indices
        for i in range(len(self.forb_tours)):
            forb_skill_comps[i] = self.forb_tours[i].skill_comp

        # 2. call algorithm
        (min_paths, min_costs, min_labels, count_labels, count_dom, tour_costs, sink_labels_network,
         no_initial_labels) = spprc_algorithm(self, forb_tour_idxs, forb_skill_comps, mu, delta, rho_gr,
                                                                        rho_le, psi, zeta_le, zeta_gr, only_best_tasks,
                                                                        best_task_cnt, t_max_le, t_max_gr,
                                                                        solve_as_dmp, node, yuan_approach)

        return min_paths, min_costs, min_labels, count_labels, count_dom, tour_costs, sink_labels_network, no_initial_labels

    def remove_forced_task(self, forced_tasks):
        """Remove nodes corresponding to forced tasks (i.e. tasks that are part of forced tours).

        Parameters
        ----------
        forced_tasks: list
            List of forced tasks
        """

        start_setup = time.time()
        # 1. get all forced tasks that can be executed by current profile
        forced_compatible_tasks = set(filter(lambda x: x in self.tasks, forced_tasks))
        # 2. remove tasks from PricingNetwork and remember their name in self.removed_tasks
        for task in forced_compatible_tasks:
            self.removed_tasks.append(task)
            self.removed_arcs += list(self.graph.in_edges(task)) + list(self.graph.out_edges(task))
            self.tasks.remove(task)
            self.graph.remove_node(task)        # remove node corresponding to task
        self.time_setup += time.time() - start_setup

    def restore_removed_tasks(self):
        """Restore previously removed tasks (e.g. when a PricingNetwork is re-used at a different branch with different
        forced tours).

        """
        start_setup = time.time()
        self.tasks.extend(self.removed_tasks)
        self.graph.add_edges_from(self.removed_arcs)    # add removed arcs, removed nodes are automatically re-added as well
        self.removed_tasks = []     # reset list of removed tasks
        self.removed_arcs = []      # and list of removed arcs
        self.time_setup += time.time() - start_setup

    def build_tour(self, label, inst):
        """Build GH_tour object based on path (i.e. solution of the SPPRC).

        Parameters
        ----------
        inst: instance_loader.Instance
            Contains all necessary instance data read from input files.
        label: workers.pricing.dynamic_programming.Label
            Label describing a tour executed by a team with the underlying pricing network's formation/profile

        Returns
        -------
        tour: workers.pricing.gh_tour.GH_tour
            Tour executed by the underlying pricing network's formation/profile
        """
        tour = GH_tour(inst.formations_w_d[self.formation_id], self.formation_id)
        tour.cost = label.tour_cost
        tour.reduced_costs = label.cost
        tour.task_reward = label.task_reward
        tour.busy_penalty = label.total_busy_penalty
        tour.leave_time = label.start_time_from_depot
        tour.quantile_return_time = label.quantile_case_finish
        tour.tour_cost = label.tour_cost
        tour.skill_comp = label.min_skill_comp
        tour.skill_comp_cnt = label.min_skill_comp_cnt
        tour.tw_viol_prob = label.tw_viol_prob.copy()
        last_finish_time = tour.leave_time
        quantile_finish_time = tour.leave_time
        # iteratively compute finish time distributions and store info
        for i in range(1, len(label.sequence)-1):
            pred = label.sequence[i - 1]
            task = label.sequence[i]
            # compute worst-case start & finish time of task
            label_t_bin_pred = self.bin_per_instant[label.median_finish_per_task[pred]]
            worst_case_start_time = max(last_finish_time + self.max_travel_times_per_bin[label_t_bin_pred][(label.sequence[i-1], task)],
                                        self.earliest_starts[task])
            quantile_start_time = max(quantile_finish_time + self.quantile_travel_times_per_bin[label_t_bin_pred][(label.sequence[i-1], task)],
                                      self.earliest_starts[task])
            worst_case_finish_time = worst_case_start_time + self.task_execution_times[task]
            quantile_finish_time = quantile_start_time + self.task_execution_times[task]
            # add task to tour and store worst-case start & finish times
            tour.tasks.append(task)
            tour.worst_case_start_time[task] = worst_case_start_time
            tour.quantile_finish_time[task] = quantile_finish_time
            last_finish_time = worst_case_finish_time      # remember finish time of last task
        tour.quantile_finish_time[label.sequence[-1]] = tour.quantile_return_time

        return tour

class Resource():
    def __init__(self, non_dom_form, forb_res):
        """Create ResourceObject for a certain node. Contains resources needed during dominance checks or to identify
        forbidden (sub)paths.
        Parameters
        ----------
        node: str
            Tail node of underlying arc, which will have this resource assigned
        non_dom_form: bool
            Indicate if the underlying pricing network's formation is dominated for node
        forb_res:
            Maps indices of forbidden tours to their length
        """
        self.forb_res = forb_res
        self.non_dom_formation = non_dom_form