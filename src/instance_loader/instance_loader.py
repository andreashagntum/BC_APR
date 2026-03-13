"""This script contains functions to load an input instance formatted as .json and parse it into the format used by
the solver.
"""

import json

def load_InstWithTimeDiscr(filepath, worker_quantile):
    """Load instance data into object of type Instance, which is the used by the solver.

    Parameters
    ----------
    filepath: str
        Filepath where instance data is located. Required format: .json
    worker_quantile: float in [0,1]
        parameter gamma from the paper, used in the workforce constraints.

    Returns
    -------
    val: float
        inst: Instance object containing all relevant instance data.
    """

    f = open(filepath)
    data = json.load(f)
    f.close()

    inst = type('Instance', (object,), {})()
    inst.tasks_per_formation = data["tasks_per_formation"]
    inst.modes = data["modes"]
    inst.depot = data["depot"]
    inst.begin_horizon = data["begin_horizon"]
    inst.end_horizon = data["end_horizon"]
    inst.earliest_start = data["earliest_start"]
    inst.latest_start = data["latest_start"]
    inst.weights = data["weights"]
    inst.earliest_finish = data["earliest_finish"]
    inst.latest_finish = data["latest_finish"]
    inst.tasks = data["tasks"]
    inst.skill_levels = data["skill_levels"]
    inst.service_level = data["service_level"]
    inst.worker_quantile = worker_quantile
    inst.latest_start_viol = data["latest_start_viol"]
    inst.latest_finish_viol = data["latest_finish_viol"]
    inst.formations = {}
    inst.task_locations = data["task_locations"]

    for f_id in data["formations"]:
        inst.formations[f_id] = {}
        for level in data["formations"][f_id]:
            inst.formations[f_id][int(level)] = data["formations"][f_id][level]
    
    inst.formations_w_d = {}
    for f_id in data["formations_w_d"]:
        inst.formations_w_d[f_id] = {}
        for level in data["formations_w_d"][f_id]:
            inst.formations_w_d[f_id][int(level)] = data["formations_w_d"][f_id][level]
    
    inst.workers = {}
    for w in data["workers"]:
        inst.workers[int(w)] = data["workers"][w]
    
    inst.workers_w_d = {}
    for w in data["workers_w_d"]:
        inst.workers_w_d[int(w)] = data["workers_w_d"][w]
    
    inst.travel_times_per_bin = {}
    for time_bin in data["travel_times"]:
        inst.travel_times_per_bin[int(time_bin)] = {}
        for i in data["travel_times"][time_bin]:
            for j in data["travel_times"][time_bin][i]:
                inst.travel_times_per_bin[int(time_bin)][(i,j)] = {int(k):v for (k,v) in data["travel_times"][time_bin][i][j].items()}  # entries are now dictionaries with keys = edges, values = prob. of travel time

    inst.bin_per_instant = {int(k): v for (k,v) in data["time_step_to_bin"].items()}
    inst.instant_per_bin = {int(b): [] for b in data["travel_times"]}
    for t in inst.bin_per_instant:
        inst.instant_per_bin[int(inst.bin_per_instant[t])].append(t)
    inst.instant_per_bin = {b: list(sorted(inst.instant_per_bin[b])) for b in inst.instant_per_bin}

    inst.bins = list(inst.travel_times_per_bin.keys())

    inst.skill_levels = []
    for level in data["skill_levels"]:
        inst.skill_levels.append(int(level))
        
    inst.instants = []
    for instant in data["instants"]:
        inst.instants.append(instant)
        
    # remove dominated modes if any
    to_be_removed = []
    for task in data["modes"]:
        for formation_id_dom in data["modes"][task]:
            for formation_id in data["modes"][task]:
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
                        if data["modes"][task][formation_id_dom] >= data["modes"][task][formation_id]:
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
                    
                    # update tasks_per_formation_with_domination
                    if task not in inst.tasks_per_formation_with_domination[formation_id_dom]:
                        inst.tasks_per_formation_with_domination[formation_id_dom].append(task) 

        
    return inst


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


    