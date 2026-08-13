"""Functions to compute the workforce for a given instance with a given workforce strength rs.
Computes the minimum and maximum workforce for a given instance. Afterwards, rescales the workforce size based on the
workforce strength rs. For additional details, see the function docstrings.
"""

def compute_workforce_rs(inst, travel_times_per_bin, rs, normalize_workforce = False, inst_wf = None):
    """Computes the workforce for a given instance inst and a workforce strength rs.
    Proceeds as follows:
    - computes a minimum and maximum workforce size using get_min_workforce and get_max_workforce
    - rescales the actual workforce via workforce strength factor rs

    Note: the minimum and maximum workforce depends on the distributions of travel times. Thus, if an instance should
    be solved using different distributions (e.g. stochastic vs. deterministic best-case), one needs to ensure that
    the available workforce is identical for both instances to ensure comparability. For this, one can set
    normalize_workforce to True and specify a baseline instance file inst_wf (e.g. the stochastic one). Then, the
    workforce size computation is done using inst_wf, independent of the actual (to-be-solved) instance inst.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    travel_times_per_bin: dict
        Maps time bins and edges to their distribution of probabilities
    rs: float in [0,1]
        Workforce strength relative to the no. of workers required to trivially solve the instance.
    normalize_workforce: bool
        True iff. workforce size calculation should be done based on a different instance file inst_wf (needed to ensure
        same workforce sizes for stochastic and deterministic instances).
    inst_wf: instance_loader.Instance
        Instance data for instance that should be used for workforce size calculation (typically: instance data from
        stochastic instance).

    Returns
    -------
    inst: instance_loader.Instance
        Input Instance object with update attributes inst.workers and inst.workers_w_d (maps skill levels to the number
        of available workers without and with downgrading, respectively, per time instant).
    """

    # 1. specify which dataframe should be used for min./max. workforce size computation and compute workforce size
    # bounds
    if normalize_workforce:
        assert inst_wf is not None
        w_min = get_min_workforce(inst_wf)
        w_max = get_max_workforce(inst_wf, travel_times_per_bin)
    else:
        w_min = get_min_workforce(inst)
        w_max = get_max_workforce(inst, travel_times_per_bin)

    # 2. scale workforce size with rs to get actual workforce size
    workers = {}
    for k in inst.skill_levels:
        workers[k] = w_min[k] + round(rs*(w_max[k] - w_min[k]))

    # 3. get workers per skill level with and without downgrading, save to inst
    for skill_level in inst.skill_levels:
        inst.workers[skill_level] = workers[skill_level]
        w_w_d = 0   # workers with downgrading
        for kk in filter(lambda x: x>=skill_level, inst.skill_levels):
            w_w_d += workers[kk]
        inst.workers_w_d[skill_level] = w_w_d

    # 4. Convert workers data to time-dependent dict
    inst.workers = {k: {t: workers[k] for t in range(inst.begin_horizon, inst.end_horizon + 1)} for k in workers}
    inst.workers_w_d = {k: {t: inst.workers_w_d[k] for t in range(inst.begin_horizon, inst.end_horizon + 1)} for k in inst.workers_w_d}

    return inst



def get_max_workforce(inst, travel_times_per_bin):
    """Calculate maximum workforce size such that a given instance inst is feasible.
    Given the fastest execution mode for each task and its earliest start time, computes the total required workforce
    per time instant. For each skill level, the workforce size is then given as the maximum workforce requirement
    at any time instant.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    travel_times_per_bin: dict
        Maps time bins and edges to their distribution of probabilities

    Returns
    -------
    max_util: dict
        Maps skill levels to no. of workers for each skill level.
    """

    # 1. get quickest mode for each task
    quickest_mode = {}
    for task in inst.tasks:
        quickest_mode[task] = min(inst.modes[task], key=inst.modes[task].get)

    # 2. initialize utilization (i.e. no of required workers) for each skill level and time instant to 0
    util_tk = {}
    for t in inst.instants:
        util_tk[t] = {}
        for k in inst.skill_levels:
            util_tk[t][k] = 0

    # 3. get no. of required workers per skill level for each time instant AND skill level, assuming each task is
    # started as early possible
    for task in inst.tasks:
        # calculate min. travel time for which service level constraint is satisfied
        distance = max([max(travel_times_per_bin[time_bin][(task, inst.depot)]) for time_bin in travel_times_per_bin])

        leave_time = inst.earliest_start[task] - distance
        return_time = inst.earliest_finish[task] + distance
        exec_instants = list(filter(lambda x: x>=leave_time and x<return_time, inst.instants))
        for t in exec_instants:
            formation = inst.formations[quickest_mode[task]]
            for k in formation:
                util_tk[t][k] += formation[k]

    # 4. get maximum no. of simultaneously required workers for each skill level over all tasks
    max_util = {}
    for k in inst.skill_levels:
        max_util[k] = 0
    for t in util_tk:
        for k in util_tk[t]:
            if max_util[k] < util_tk[t][k]:
                max_util[k] = util_tk[t][k]
            
    return max_util

def get_min_workforce(inst):
    """Calculate minimum workforce size such that a given instance inst is potentially feasible.
    Logic is analogous to get_max_workforce, but slowest modes are used instead. Also, no assumption on task
    start times are made.
    Bound is (almost) never tight, i.e., if the workforce computed by this function is provided to the algorithm,
    the problem is (almost) always infeasible.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.

    Returns
    -------
    max_util: dict
        Maps skill levels to no. of workers for each skill level.
    """

    # 1. get slowest mode for each task to be performed
    slowest_mode = {}
    for task in inst.tasks:
        slowest_mode[task] = max(inst.modes[task],key=inst.modes[task].get)

    # 2. initialize utilization (i.e. no of required workers) for each skill level to 0
    max_util = {}
    for k in inst.skill_levels:
        max_util[k] = 0

    # 3. sum up the no. of required workers per skill level for each skill level over slowest modes for all tasks
    for task in inst.tasks:
        formation = inst.formations[slowest_mode[task]]
        for k in formation:
            if max_util[k] < formation[k]:
                max_util[k] = formation[k]
                
    return max_util
