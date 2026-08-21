"""This script contains functions to load an input instance formatted as .json and parse it into the format used by
the solver.
"""

import json
from tqdm import tqdm
import copy

def convert_to_distributions(data):
    """Convert old data format into new format by converting deterministic time windows and service times into
    distributions.

    Parameters
    ----------
    data: dict
       Instance JSON read from disk

    Returns
    -------
    data: dict
       Instance JSON with datastructures adjusted to match the new format
    """

    data["scheduled_earliest_start"] = data["earliest_start"]
    data["scheduled_latest_finish"] = data["latest_finish"]


    data["earliest_start"] = {task: {data["earliest_start"][task]: 1} for task in data["earliest_start"]}

    data["modes"] = {task: {f: {data["modes"][task][f]: 1} for f in data["modes"][task]} for task in data["modes"]}


    return data


def load_InstWithTimeDiscr(filepath, worker_quantile, estimator_fn, monte_carlo = False, sample_cnt = 1,
                           rng = None, t_len = None, tt_data_path = None, inst_type = None,
                           max_twviol = None, service_level = None):
    """Load instance data into object of type Instance, which is the used by the solver.

    Parameters
    ----------
    filepath: str
        Filepath where instance data is located. Required format: .json
    worker_quantile: float
        parameter gamma from the paper, used in the workforce constraints.
    estimator_fn: function
        Function used to estimate service times and time windows (e.g. min, max, mean)
    monte_carlo: bool
        Indicates if the instance is intended to be solved using the monte-carlo approach
    sample_cnt: int
        Number of samples requested
    rng: np.random.RandomState.Generator
        RNG object for sampling
    t_len: int or None
        Time length per instance. Only used if instance JSON contains no instantLength key
    tt_data_path: str
       Path where centralized travel time JSON is stored. Only used in instance JSON contains no travel_time key.
    inst_type: 'new' OR 'classic'
       Indicates if the input instances uses the classic format (of instances used for Hagn et al. (2026)) or the new
       format (with stochastic service times and time windows, and several additional auxiliary data)
    max_twviol: int or NoneType
       If passed, overrides the max_twviol key of the instance JSON file. If inst_type == "classic", this value MUST be int
    service_level: float or NoneType
       If passed, overrides the service_level key of the instance JSON file
    Returns
    -------
    val: float
        inst: Instance object containing all relevant instance data.
    travel_times: dict
        Dictionary containing travel times per time bin and edge
    """


    f = open(filepath)
    data = json.load(f, object_hook=keys_to_int)
    f.close()

    if inst_type == "classic":
        data = convert_to_distributions(data)

    data["max_twviol"] = max_twviol if max_twviol is not None else data["max_twviol"]
    data["service_level"] = service_level if max_twviol is not None else data["service_level"]


    inst = type('Instance', (object,), {})()
    inst.tasks_per_formation = data["tasks_per_formation"]
    inst.depot = data["depot"]
    inst.begin_horizon = data["begin_horizon"]
    inst.end_horizon = data["end_horizon"]
    inst.weights = data["weights"]
    inst.tasks = data["tasks"]
    inst.tasks_from_prev_segs = [] # new: tasks carried over from previous segments: no-reoptimization permitted for these tasks as they are already executed
    inst.fixed_finish_times = {} # new: fixed finish times for tasks carried over from previous segments: during pricing, cost will be computed deterministically
    inst.all_tasks = inst.tasks + inst.tasks_from_prev_segs
    inst.active_tours_from_prev_segs = [] # tours from previous segment that are still active (tasks can be enqueued for these tours, or they can be sent to the depot)
    inst.skill_levels = data["skill_levels"]
    inst.service_level = data["service_level"]
    inst.worker_quantile = worker_quantile
    if "instantLength" not in data:
        data["instantLength"] = t_len
    inst.always_feas_edges = [] # list of time bins and edges for which chance constr./LFv feas. check will be skipped in the next BPC&S iteration

    # compute sampled service times and time windows
    inst.monte_carlo = True
    inst.multi_sampled_modes = [{} for _ in range(sample_cnt)]
    inst.multi_sampled_modes_with_domination = [{} for _ in range(sample_cnt)]
    inst.multi_sampled_earliest_start = [{} for _ in range(sample_cnt)]
    inst.multi_sampled_latest_finish = [{} for _ in range(sample_cnt)]
    inst.multi_sampled_latest_finish_viol = [{} for _ in range(sample_cnt)]
    for i in range(sample_cnt):
        for task in data["tasks"]:
            # sample task execution times
            inst.multi_sampled_modes[i][task] = {}
            inst.multi_sampled_modes_with_domination[i][task] = {}
            for f in data["modes"][task]:
                items = list(data["modes"][task][f].items())
                if rng:
                    inst.multi_sampled_modes[i][task][f] = rng.choice([k for (k, _) in items], p = [p for (_, p) in items])
                else: # fallback for old instance format
                    inst.multi_sampled_modes[i][task][f] = items[0][0]
                # for all other dominated modes, values will be set at runtime
                inst.multi_sampled_modes_with_domination[i][task][f] = inst.multi_sampled_modes[i][task][f]
            # sample time window
            items_tw = list(data["earliest_start"][task].items())
            if rng:
                inst.multi_sampled_earliest_start[i][task] = rng.choice([k for (k, _) in items_tw], p = [p for (_, p) in items_tw])
            else: # fallback for old instance format
                inst.multi_sampled_earliest_start[i][task] = items_tw[0][0]
            # latest finish: depends on earliest start + scheduled time window length
            tw_len = data["scheduled_latest_finish"][task] - data["scheduled_earliest_start"][task]
            inst.multi_sampled_latest_finish[i][task] = inst.multi_sampled_earliest_start[i][task] + tw_len
            inst.multi_sampled_latest_finish_viol[i][task] = inst.multi_sampled_latest_finish[i][task] + data["max_twviol"]
            # Note: we do not explicitly computed earliest_finish and latest_start because sampling is only used for MC
            # tour evaluation, which in turn only cares about earliest_start and latest_finish
    inst.sampled_modes = copy.deepcopy(inst.multi_sampled_modes[0])
    inst.sampled_modes_with_domination = copy.deepcopy(inst.multi_sampled_modes_with_domination[0])
    inst.sampled_earliest_start = copy.deepcopy(inst.multi_sampled_earliest_start[0])
    inst.sampled_latest_finish = copy.deepcopy(inst.multi_sampled_latest_finish[0])
    inst.sampled_latest_finish_viol =copy.deepcopy(inst.multi_sampled_latest_finish_viol[0])

    inst.monte_carlo = monte_carlo

    # convert task execution times to deterministic values
    inst.modes = {task: {mode: estimator_fn(data["modes"][task][mode]) for mode in data["modes"][task]} for task in data["modes"]}

    # handle time windows: we sample based on the given estimator_fn
    inst.earliest_start = {}
    inst.latest_finish = {}
    inst.latest_finish_viol = {}
    for task in inst.tasks:
        tw_len = data["scheduled_latest_finish"][task] - data["scheduled_earliest_start"][task]
        inst.earliest_start[task] = estimator_fn(data["earliest_start"][task])
        inst.latest_finish[task] = inst.earliest_start[task] + tw_len
        inst.latest_finish_viol[task] = inst.latest_finish[task] + data["max_twviol"]
        # earliest_finish, latest_start, and latest_start_viol depend on earliest_start + min_execution_time
        # and on latest_finish - min_execution_time, respectively, and will thus be computed at the end of this script
        # (after dominated mode removal)

    inst.formations = {}
    inst.task_locations = data["task_locations"]

    for f_id in data["formations"]:
        inst.formations[f_id] = {}
        for level in data["formations"][f_id]:
            inst.formations[f_id][level] = data["formations"][f_id][level]
    
    inst.formations_w_d = {}
    for f_id in data["formations_w_d"]:
        inst.formations_w_d[f_id] = {}
        for level in data["formations_w_d"][f_id]:
            inst.formations_w_d[f_id][level] = data["formations_w_d"][f_id][level]
    
    inst.workers = {}
    for w in data["workers"]:
        inst.workers[w] = data["workers"][w]
    
    inst.workers_w_d = {}
    for w in data["workers_w_d"]:
        inst.workers_w_d[int(w)] = data["workers_w_d"][w]

    # if travel_times data is contained in JSON: directly read it
    if "travel_times" in data:
        travel_times_per_bin = {}
        for time_bin in data["travel_times"]:
            travel_times_per_bin[time_bin] = {}
            for i in data["travel_times"][time_bin]:
                for j in data["travel_times"][time_bin][i]:
                    travel_times_per_bin[time_bin][(i,j)] = {k: v for (k,v) in data["travel_times"][time_bin][i][j].items()}  # entries are now dictionaries with keys = edges, values = prob. of travel time
    # else: construct it from central JSON file
    else:
        with open(f"{tt_data_path}/stoch_travel_times_bin_len{data["bin_minutes"]}_instantLength{data["instantLength"]}.json",
                "r") as f:
            stoch_tts = json.load(f, object_hook=keys_to_int)
            stoch_tts = {time_bin: d for (time_bin, d) in stoch_tts}

        travel_times_per_bin = {}
        # need to rescale if instant length is not equal to 1
        if data["instantLength"] != 1:
            for time_bin in tqdm(stoch_tts, "Travel time bins constructed:"):
                travel_times_per_bin[time_bin] = {}
                for i in data["travel_times"][time_bin]:
                    gate_i = data["task_locations"][i]
                    for j in data["travel_times"][time_bin][i]:
                        gate_j = data["task_locations"][j]
                        data["travel_times"][time_bin][(i, j)] = clip_and_rescale_distribution(stoch_tts[time_bin][min(gate_i, gate_j)][max(gate_i, gate_j)],
                                                                                              data["instantLength"])

        else:
            nodes = data["task_locations"]
            for time_bin in tqdm(stoch_tts, "Travel time bins constructed:"):
                travel_times_per_bin[time_bin] = {}
                for i in nodes:
                    gate_i = int(data["task_locations"][i])
                    for j in nodes:
                        gate_j = int(data["task_locations"][j])
                        travel_times_per_bin[time_bin][(i, j)] = stoch_tts[time_bin][min(gate_i, gate_j)][max(gate_i, gate_j)]

    inst.bin_per_instant = {k: v for (k,v) in data["time_step_to_bin"].items()}
    # add bin data if missing
    for t in data["instants"]:
        if t not in inst.bin_per_instant:
            next_smaller_t = max([tt for tt in inst.bin_per_instant if tt <= t])
            if next_smaller_t:
                inst.bin_per_instant[t] = inst.bin_per_instant[next_smaller_t]
            else:
                next_larger_t = min([tt for tt in inst.bin_per_instant if tt > t])
                inst.bin_per_instant[t] = inst.bin_per_instant[next_larger_t]
    # reverse mapping
    inst.instant_per_bin = {b: [] for b in travel_times_per_bin}
    for t in inst.bin_per_instant:
        inst.instant_per_bin[inst.bin_per_instant[t]].append(t)
    inst.instant_per_bin = {b: list(sorted(inst.instant_per_bin[b])) for b in inst.instant_per_bin}


    inst.bins = list(travel_times_per_bin.keys())
    inst.sampled_tts = {b: {} for b in inst.bins}

    inst.skill_levels = []
    for level in data["skill_levels"]:
        inst.skill_levels.append(level)
        
    inst.instants = []
    for instant in data["instants"]:
        inst.instants.append(instant)

    # store last instant of day: depends on instantLength and will be used downstream to identify bins for times after
    # midnight
    if abs(int(24 * 60 / data["instantLength"]) - (24 * 60 / data["instantLength"])):
        raise Exception(f"instantLength that is not a divisor of 1440 not supported")
    inst.last_inst_of_day = int(24 * 60 / data["instantLength"])

        
    # remove dominated modes if any
    to_be_removed = []
    for task in inst.modes:
        for formation_id_dom in inst.modes[task]:
            for formation_id in inst.modes[task]:
                if formation_id_dom != formation_id:
                    dominated = True
                    min_skill_formation = min(inst.formations_w_d[formation_id].keys())
                    min_skill_formation_dom = min(inst.formations_w_d[formation_id_dom].keys())
                    if min_skill_formation < min_skill_formation_dom:   # Just to make sure formation_id_dom_w_d contains min_skill_formation
                        for k in range(min_skill_formation, min_skill_formation_dom):
                            inst.formations_w_d[formation_id_dom][k] =  inst.formations_w_d[formation_id_dom][min_skill_formation_dom]
                    for k in inst.formations_w_d[formation_id]:
                        if k not in inst.formations_w_d[formation_id_dom] or inst.formations_w_d[formation_id_dom][k] < inst.formations_w_d[formation_id][k]:
                            dominated = False
                            break
                    if dominated:
                        if inst.modes[task][formation_id_dom] >= inst.modes[task][formation_id]:
                            to_be_removed.append((task,formation_id_dom))
    for task__formation_id in to_be_removed:
        del inst.modes[task__formation_id[0]][task__formation_id[1]]
        inst.tasks_per_formation[task__formation_id[1]].remove(task__formation_id[0])               

    # calculate formation domination 
    inst.tasks_per_formation_with_domination = {}
    for formation_id in inst.formations:
        inst.tasks_per_formation_with_domination[formation_id] = []
    inst.modes_with_domination = {}
    for task in inst.tasks:
        inst.modes_with_domination[task] = {}
    for formation_id in inst.formations:
        for formation_id_dom in inst.formations:

            dominated = True
            # first check the nr of workers is the same
            min_skill_formation = min(inst.formations_w_d[formation_id].keys())
            min_skill_formation_dom = min(inst.formations_w_d[formation_id_dom].keys())
            if inst.formations_w_d[formation_id][min_skill_formation] != inst.formations_w_d[formation_id_dom][min_skill_formation_dom]:
                dominated = False
                continue
            # now check the skills
            if min_skill_formation < min_skill_formation_dom:   # Just to make sure formation_id_dom_w_d contains min_skill_formation
                for k in range(min_skill_formation, min_skill_formation_dom):
                    inst.formations_w_d[formation_id_dom][k] =  inst.formations_w_d[formation_id_dom][min_skill_formation_dom]
            for k in inst.formations_w_d[formation_id]:
                if k not in inst.formations_w_d[formation_id_dom] or inst.formations_w_d[formation_id_dom][k] < inst.formations_w_d[formation_id][k]:
                    dominated = False
                    break
            if dominated:
                for task in inst.tasks_per_formation[formation_id]:
                    # update modes_with_domination
                    if formation_id_dom not in inst.modes_with_domination[task]:
                        inst.modes_with_domination[task][formation_id_dom] = inst.modes[task][formation_id]
                        # transfer sampled service time
                        inst.sampled_modes_with_domination[task][formation_id_dom] = inst.sampled_modes[task][formation_id]
                        for i in range(sample_cnt):
                            inst.multi_sampled_modes_with_domination[i][task][formation_id_dom] = inst.multi_sampled_modes_with_domination[i][task][formation_id]

                    # update tasks_per_formation_with_domination
                    if task not in inst.tasks_per_formation_with_domination[formation_id_dom]:
                        inst.tasks_per_formation_with_domination[formation_id_dom].append(task) 

    # define remaining time window structures: earliest_start, latest_start, and latest_start_viol
    inst.earliest_finish = {task: inst.earliest_start[task] + min(inst.modes[task].values()) for task in data["tasks"]}
    inst.latest_start = {task: inst.latest_finish[task] - min(inst.modes[task].values()) for task in data["tasks"]}
    inst.latest_start_viol = {task: inst.latest_start[task] + data["max_twviol"] for task in data["tasks"]}
    # also sample earliest finish times, as we will need it for net cost computation
    inst.sampled_earliest_finish = {task: inst.sampled_earliest_start[task] + min(inst.sampled_modes[task].values())
                                    for task in data["tasks"]}
    inst.sampled_latest_start = {task: inst.sampled_latest_finish[task] - min(inst.sampled_modes[task].values())
                                 for task in data["tasks"]}
    inst.sampled_latest_start_viol = {task: inst.sampled_latest_finish_viol[task] - min(inst.sampled_modes[task].values())
                                      for task in data["tasks"]}
    # transfer to multi-sample
    inst.multi_sampled_earliest_finish = [{task: inst.multi_sampled_earliest_start[i][task] +
                                                 min(inst.multi_sampled_modes[i][task].values())
                                           for task in data["tasks"]} for i in range(sample_cnt)]
    inst.multi_sampled_latest_start = [{task: inst.multi_sampled_latest_finish[i][task] -
                                              min(inst.multi_sampled_modes[i][task].values()) for task in data["tasks"]}
                                       for i in range(sample_cnt)]
    inst.multi_sampled_latest_start_viol = [{task: inst.multi_sampled_latest_finish_viol[i][task] -
                                              min(inst.multi_sampled_modes[i][task].values()) for task in data["tasks"]}
                                       for i in range(sample_cnt)]

    return inst, travel_times_per_bin

def keys_to_int(d):
    """Helper function that automatically parses string-formatted integer keys into integers.
    Also handles string-formatted floats and converts them to the corresponding float value.
    Note: as of August 2026, we do not use float-indexed keys, but we still provide this functionality for the future.
    """
    new_d = {}
    for k, v in d.items():
        try:
            key = int(k)
        except ValueError:
            try:
                f = float(k)
                key = int(f) if f.is_integer() else f
            except ValueError:
                key = k
        new_d[key] = v
    return new_d


def add_travel_times_per_bin_restricted(inst, travel_times_per_bin, time_bins):
    """Add travel time data to instance. Currently a shallow copy, as we never mutate this object.
    Only copies travel times for bins in time_bins, since when instances are solved via rolling horizon, only a subset
    of time bins is relevant at a time.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    travel_times_per_bin: dict
        Maps time bins and edges to their distribution of probabilities
    time_bins: list[int]
        List of all time bins whose distributions should be copied.

    """
    # Nested shallow copy
    inst.travel_times_per_bin = {time_bin: {edge: dict(travel_times_per_bin[time_bin][edge]) for edge in travel_times_per_bin[time_bin]}
                                 for time_bin in time_bins}


    return

def get_min_value(inst):
    """Computes the minimum possible objective value of an instance inst, even if the solution is infeasible.
    This value is used to offset the objective value of a feasible solution for inst, in order to be able to compute
    meaningful gaps.
    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.

    Returns
    -------
    val: float
        minimum objective value of any feasible solution for inst.
    """
    # compute lower bound as sum(earliest_finish[task]*task_weight for all tasks)
    val = 0
    for task in inst.tasks:
        val += inst.earliest_finish[task] * inst.weights[task]
    return val

def clip_and_rescale_distribution(distr, instant_length=1, eps_prob=1e-2):
    """Clip and rescales distribution.
    First, all travel time observations are rescaled to the nearest multiple of instant_length and rounded to few
    digits.
    Then, removes all entries whose probabilities are below eps_prob. Afterwards, fills the dictionary
    with zero-probability entries for all observations between the lowest and highest value with missing key in distr.
    Finally, rescales all probabilities to ensure they sum up to 1 and rounds them to a given number of digits.

    Note: This function is only used when a centralized travel time JSON is provided. The latter results from a KDE
    approach to estimate travel time distributions, which by construction can produce arbitrarily large or small values
    (since they typically use real-valued continuous functions as kernel functions). Therefore, steps such as quantile-
    based pruning become necessary. If travel time data is constructed using the approach by Hagn et al. (2026), this
    step is not necessary. For such instances, it is better to directly store the travel times in the instance JSON.

    Parameters
    ----------
    distr: dict
       Maps travel time values to their float-formatted probabilities
    instant_length: int
       Length per time instant (in minutes)
    eps_prob: float
       Controls the quantile levels at which the distribution tails should be cut off.

    Returns
    -------
    distr_new: dict
       Clipped and rescaled distribution
    """
    # 1. Rescale to instant_length
    distr_new = {}
    for (k, v) in distr.items():
        new_k = int(round(k / instant_length, 0))
        if new_k not in distr_new:
            distr_new[new_k] = 0
        distr_new[new_k] += v

    # 2. Get bounds
    lb = get_lower_quantile(distr_new, eps_prob)
    ub = get_upper_quantile(distr_new, 1 - eps_prob)

    # 3. Remove anything out of bounds
    distr_new = {int(k): float(v) for (k, v) in distr_new.items() if (k >= lb and k <= ub)}

    # 4. Add zero probability for missing values
    for k in range(min(distr_new), max(distr_new)):
        if k not in distr_new:
            distr_new[k] = 0

    # 5. Sort and re-create dict
    distr_new = dict(sorted(distr_new.items(), key=lambda x: x[0]))

    # 6. Rescale to ensure probabilities sum up to one
    prob_sum = sum(distr_new.values())
    distr_new = {k: v / prob_sum for (k, v) in distr_new.items()}

    # 7. Round all probabilities to reduce storage footprint
    distr_new = round_distribution(distr_new, digits=3)

    return distr_new

def keys_to_int(d):
    """Helper function that automatically parses string-formatted integer keys into integers.
    Also handles string-formatted floats and converts them to the corresponding float value.
    """
    new_d = {}
    for k, v in d.items():
        try:
            key = int(k)
        except ValueError:
            try:
                f = float(k)
                key = int(f) if f.is_integer() else f
            except ValueError:
                key = k
        new_d[key] = v
    return new_d

def get_upper_quantile(d, q):
    subsum = 1
    last = max(d.keys())
    for k in sorted(d.keys(), reverse=True):
        if subsum - d[k] < q:
            return last
        subsum -= d[k]
        last = k
    return last


def get_lower_quantile(d, q):
    subsum = 1
    for k in sorted(d.keys(), reverse=True):
        if subsum - d[k] < q:
            return k
        subsum -= d[k]
    return k


def round_distribution(distr, digits=3):
    """Round all probabilities. Remaining probability mass is assigned to the first key (i.e., the lowest value).

    Parameters
    ----------
    distr: dict
       Maps travel time values to their float-formatted probabilities

    Returns
    -------
    floored_scaled: dict
       Rounded distribution
    """

    # 1. Scale probabilities temporarily
    scale = 10 ** digits
    keys = list(distr.keys())
    scaled = {k: distr[k] * scale for k in keys}

    # 2. Floor everything first and compute remainder
    floored = {k: int(scaled[k]) for k in keys}
    remainder = {k: scaled[k] - floored[k] for k in keys}

    # 3. Compute how many units are we short of the target total
    total_floor = sum(floored.values())
    missing = round(scale - total_floor)  # should be a small non-negative int

    #4. Give the +1 units to the entries with the largest remainders
    for k in sorted(keys, key=lambda k: remainder[k], reverse=True)[:missing]:
        floored[k] += 1

    # 5. Safety-check if we got any negative values or values don't sum to 1
    floored_scaled = {k: floored[k] / scale for k in keys}
    tol = 1e-4
    if min(floored_scaled.values()) < 0 or sum(floored_scaled.values()) > 1 + tol or sum(
            floored_scaled.values()) < 1 - tol:
        raise Exception(f"Negative values or distribution doesn't sum to 1")

    # 6. Remove head and tail if distribution starts/ends with zero-values
    first_nonzero_prob = min([k for (k, v) in floored_scaled.items() if v > tol])
    last_nonzero_prob = max([k for (k, v) in floored_scaled.items() if v > tol])

    # 7. Filter and sort ascending w.r.t keys
    floored_scaled = {k: v for (k, v) in sorted(floored_scaled.items(), key = lambda x: x[0])
                      if (k >= first_nonzero_prob) and (k <= last_nonzero_prob)}

    return floored_scaled
