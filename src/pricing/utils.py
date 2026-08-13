"""This file contains several utility functions used during pricing.
"""
import itertools
from config.config import alpha_tol

def get_all_skill_comps(inst, formation_id):
    """Get all possible skill compositions for a given formation.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    formation_id: str
        String-formatted ID of the given formation (obtained from inst.formations)

    Returns
    -------
    feasible_comps: list
        List of all possible skill compositions with information on the actual qualifications of each worker for each
        skill required by formation_id.
    feasible_comps_cnt: list
        Similar to feasible_comps, but information is aggregated on a skill level-basis.
    feasible_comps_cnt_ids: dict
        Maps strings akin to inst.formations keys to the corresponding entry in feasible_comps_cnt
    feasible_comps_ids: dict
        Maps strings akin to inst.formations keys to the corresponding entry in feasible_comps
    """
    # 1. standardize formation by filling worker requirement for skill levels that are not needed by formation_id with 0
    formation_dict = inst.formations[formation_id]
    formation_dict_new = {}
    for k in inst.workers:
        if k not in formation_dict:
            formation_dict_new[k] = 0
        else:
            formation_dict_new[k] = formation_dict[k]
    formation_dict = formation_dict_new.copy()
    # create an inverted version
    formation_dict_inv = {}
    for i in range(len(formation_dict.keys())):
        key = list(formation_dict.keys())[-i-1]
        formation_dict_inv[key] = formation_dict[key]

    # 2. get all possible skill compositions for each skill level k
    all_tuples = {}
    for k in formation_dict:
        iter_list = [i for i in range(formation_dict[k]+1)]
        all_tuples_k_temp = list(itertools.combinations_with_replacement(iter_list, len([j for j in formation_dict_inv if j >= k])))
        all_tuples[k] = []
        for tup in all_tuples_k_temp:
            all_tuples[k] += list(set(itertools.permutations(tup)))

    # 3. filter out all impossible comps. (i.e. compositions using too many workers of at least one skill level)
    feasible_comps = []
    feasible_comps_cnt = []
    feasible_comps_cnt_ids = {}
    feasible_comps_ids = {}
    all_tuples_list = [p for p in itertools.product(*all_tuples.values())]
    formation_dict_keys = list(formation_dict.keys())
    for tup in all_tuples_list:
        workers_per_level = {}      # keys: skill levels k, values: no. of workers with actual skill lever=k that are required
        skill_comp = {}
        for k in formation_dict:
            skill_comp[k] = {}
            for kk in formation_dict:
                if kk < k:
                    continue
                skill_comp[k][kk] = 0
        for k in inst.workers:      # also contains info about unneeded skill level (makes it easier to calculate red. costs)
            workers_per_level[k] = 0

        for i in range(len(tup)):    # for each skill level: sum required workers per skill level
            k = formation_dict_keys[i]
            for j in range(len(tup[i])):
                kk = formation_dict_keys[j + i]
                workers_per_level[kk] += tup[i][j]
                skill_comp[k][kk] += tup[i][j]

        # 4. check if formation is feasible
        is_feasible = True
        for k in workers_per_level:
            # 4.1 for each skill level k: comp. needs at most inst.workers[k] many workers w/ skill level k
            if workers_per_level[k] > max(inst.workers[k].values()): # if available workers suffice for any t, we create the skill comp
                is_feasible = False
                break
            # 4.2 need to exactly satisfy the required workforce per skill level
            if sum(skill_comp[k].values()) != formation_dict[k]:
                is_feasible = False
                break

        # 5. for each possible skill compositions count: select only one skill composition
        if is_feasible and workers_per_level not in feasible_comps_cnt:
            feasible_comps.append(skill_comp)
            feasible_comps_cnt.append(workers_per_level)
            id_string = "s_"    # string representing the skill composition
            for k in workers_per_level:
                id_string += f"{k}:{workers_per_level[k]},"
            id_string = id_string.rstrip(",")
            feasible_comps_cnt_ids[id_string] = workers_per_level
            feasible_comps_ids[id_string] = skill_comp

    return feasible_comps, feasible_comps_cnt, feasible_comps_cnt_ids, feasible_comps_ids

def find_alpha_quantile(start_time_cdf, alpha):
    """Find alpha quantile in a given CDF. Usually runs faster than bisect_left, as our CDFs tend to have very few values
    and the desired alpha quantile tends to be at the end of the dictionaries' value list.
    Can be used for any type of CDF, but is typically only used for start time CDFs.

    Parameters
    ----------
    start_time_cdf: dict
        Cumulative distribution function (CDF) of start times at current last task
    alpha: float in [0,1]
        target quantile

    Returns
    -------
    k: float
        largest alpha-quantile of start_time_cdf
    """
    assert len(start_time_cdf) > 0 # can not compute quantiles for empty CDFs

    for k in reversed(start_time_cdf.keys()):
        if start_time_cdf[k] <= alpha - alpha_tol:
            return k
    return k        # return first key if all values > alpha

def find_alpha_quantile_pmf(start_time_pmf, alpha):
    """Similar to find_alpha_quantile, but for PMFs. Assumes start_time_pmf is sorted ascending.
    Can be used for any type of PMF, but is typically only used for start time PMFs.

    Parameters
    ----------
    start_time_pmf: dict
        Probability mass function (PMF) of start times at current last task
    alpha: float in [0,1]
        target quantile

    Returns
    -------
    k: float
        largest alpha-quantile of start_time_pmf
    """
    assert len(start_time_pmf) > 0 # can not compute quantiles for empty PMFs

    cumulative = 0.0
    for key in start_time_pmf:
        cumulative += start_time_pmf[key]
        if cumulative >= alpha - alpha_tol:
            return key

    # fallback if PMF does not exactly sum to 1 (needed when alpha=1)
    return key

def get_busy_penalty(formation, delta, t_from, t_to, skill_comp_cnt, solve_as_dmp):
    """Calculate reduced cost penalty for keeping workers occupied in time interval [t_from, t_to).
    If solve_as_dmp = False, i.e., the current node is solved as an AMP, workforce requirements are derived from
    'formation'. Otherwise, 'skill_comp_cnt' is used.

    Parameters
    ----------
    formation: dict
        Maps skill levels to the number of required workers (including downgrading).
    delta: dict
        Maps tuples (skill_level, time_instant) to the dual value of the corresponding workforce constraint.
    t_from: int
        Time instant that marks the beginning of the considered time interval
    t_to: int
        End of the considered time interval
    skill_comp_cnt: dict
        Maps skill levels to the number of required workers (from skill composition)
    solve_as_dmp: bool
        Indicates if the current node is solved as a DMP (True) or an AMP (False).

    Returns
    -------
    penalty: float
        Total dual cost of workforce constraints in [t_from, t_to) for a team with formation 'formation' or skill
        composition 'skill_comp_cnt'.
    """

    # 1. if node is solved as an AMP: use team formation (i.e. profile) information
    if not solve_as_dmp:
        penalty = 0
        entries_delta = {}
        for k in formation:
            entries_delta[k] = []
        for (k,t) in delta:
            if t >= t_from and t < t_to and k in formation:
                entries_delta[k].append(t)
        for k in formation:
            penalty += formation[k] * sum([delta[(k,t)] for t in entries_delta[k]])

    # 2. else: use skill composition information
    else:
        penalty = 0
        entries_delta = {}
        for k in skill_comp_cnt:
            entries_delta[k] = []
        for (k,t) in delta:
            if t >= t_from and t < t_to and k in skill_comp_cnt:
                entries_delta[k].append(t)
        for k in skill_comp_cnt:
            penalty += skill_comp_cnt[k] * sum([delta[(k,t)] for t in entries_delta[k]])

    return penalty

def get_gomory_cut_coeff_increase(formation, u_kt, t_from, t_to):
    """Calculate increase in gomory cut coefficient for a given gomory cut. Used to compute the gomory cut coefficient
    of a subtour after extending it along an arc that extends the tour's activity period to [t_from, t_to).

    Note: as stated in Hagn et al. (2026), CGCs are always based on formation/profile information, not on skill compositions.
    Even when a node is solved as a DMP, CGCs based on profiles remain valid.

    Parameters
    ----------
    formation: dict
        Maps skill levels to the number of required workers (including downgrading).
    u_kt: dict
        Maps tuples (skill_level, time_instant) to the gomory cut coefficient of the respective workforce constraint.
    t_from: int
        Time instant that marks the beginning of the considered time interval
    t_to: int
        End of the considered time interval

    Returns
    -------
    coeff: float
        Current gomory cut coefficient of the subtour (not floored yet)
    """

    coeff = 0
    entries_u_kt = {}
    for k in formation:
        entries_u_kt[k] = []
    for (k, t) in u_kt:
        if t >= t_from and t < t_to and k in formation:
            entries_u_kt[k].append(t)
    for k in formation:
        coeff += formation[k] * sum([u_kt[(k, t)] for t in entries_u_kt[k]])

    return coeff

def get_branching_penalty(rho_gr, rho_le, t_from, t_to):
    """Calculate reduced cost penalty corresponding to constraints derived from branching on the number of tours at
    a specific time.

    Parameters
    ----------
    rho_gr: dict
        Maps >= tour count constraints to their dual values
    rho_le: dict
        Maps >= tour count constraints to their dual values
    t_from: int
        Time instant that marks the beginning of the considered time interval
    t_to: int
        End of the considered time interval

    Returns
    -------
    penalty: float
        Total dual cost of workforce constraints in [t_from, t_to) for a team with formation 'formation' or skill
        composition 'skill_comp_cnt'.
    """

    penalty = (-sum([rho_gr[t] for t in rho_gr if t >= t_from and t < t_to]) +
               sum([rho_le[t] for t in rho_le if t >= t_from and t < t_to]))

    return penalty
