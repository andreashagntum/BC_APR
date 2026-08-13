"""Utility functions used for solving instances with a rolling horizon approach.
"""
from src.instance_loader.instance_loader import add_travel_times_per_bin_restricted

def get_segments(inst, segment_length):
    """Segment time horizon into segments of length 'segment_length'. Last segment might be shorter than the others.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    segment_length: int
        Length of each segment (in time steps) during rolling horizon separation

    Returns
    -------
    th_segments: list[(int, int)]
       Tuples indicating the start and end of a segment (both time instants included)
    """
    th_segments = []
    start = inst.begin_horizon
    while start <= inst.end_horizon:
        end = min(start + segment_length - 1, inst.end_horizon)
        th_segments.append((start, end))
        start += segment_length

    return th_segments


def adjust_inst_to_segment(inst, th_segment, travel_times_per_bin):
    """Adjust a (deepcopied version of the input) inst object according to a passed th_segment. The following changes are made:

    - inst.tasks is restricted to the set of tasks whose earliest start lie in th_segment
    - inst.begin_horizon and inst.end_horizon are adjusted based on task's earliest starts and latest extended finish times
    - unused formations are removed
    - time window-related values are restricted to the task subset
    - instants, mappings of bins to instants (and vice versa) are restricted to the considered segment
    - travel times are restricted to the considered segment

    Parameters
    ---------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    th_segment: tuple[int, int]
        First and last instant of the considered segment
    travel_times_per_bin: dict
        Maps time bins and edges to their distribution of probabilities

    """
    seg_start, seg_end = th_segment

    # 1. Get all tasks in considered segment
    inst.tasks = [task for task in inst.tasks if (inst.earliest_start[task] >= seg_start and inst.earliest_start[task] < seg_end)]

    # 2. Restrict all sets relating to tasks
    # 2.1 Time windows
    inst.earliest_start = {task: inst.earliest_start[task] for task in inst.tasks}
    inst.latest_start = {task: inst.latest_start[task] for task in inst.tasks}
    inst.earliest_finish = {task: inst.earliest_finish[task] for task in inst.tasks}
    inst.latest_finish = {task: inst.latest_finish[task] for task in inst.tasks}
    inst.latest_start_viol = {task: inst.latest_start_viol[task] for task in inst.tasks}
    inst.latest_finish_viol = {task: inst.latest_finish_viol[task] for task in inst.tasks}

    # 2.2 Modes, weights, task locations, and tasks per formation
    inst.modes = {task: inst.modes[task] for task in inst.tasks}
    inst.modes_with_domination = {task: inst.modes_with_domination[task] for task in inst.tasks}
    inst.weights = {task: inst.weights[task] for task in inst.tasks}
    inst.task_locations = {task: inst.task_locations[task] for task in inst.tasks}

    # 2.3 Tasks per formation: formations with no tasks will be dropped
    inst.tasks_per_formation = {f: list(set(inst.tasks_per_formation[f]).intersection(set(inst.tasks))) for f in inst.tasks_per_formation}
    to_drop = [f for f in inst.tasks_per_formation if not inst.tasks_per_formation]
    for f in to_drop:
        del inst.tasks_per_formation[f]

    # 2.4 Same for tasks per formation with domination
    inst.tasks_per_formation_with_domination = {f: list(set(inst.tasks_per_formation_with_domination[f]).intersection(set(inst.tasks)))
                                                for f in inst.tasks_per_formation_with_domination}
    to_drop_with_domination = [f for f in inst.tasks_per_formation_with_domination if not inst.tasks_per_formation_with_domination]
    for f in to_drop_with_domination:
        del inst.tasks_per_formation_with_domination[f]

    # 2.5 Drop formations entry for all formations that are not suitable for any task, neither normally nor with domination
    for f in set(to_drop).intersection(set(to_drop_with_domination)):
        del inst.formations[f]
        del inst.formations_w_d[f]


    # 3. Restrict time horizon
    # 3.1 Get earliest possible leave time for any task: this will be the horizon begin
    earliest_leaves = []
    for task in inst.tasks:
        # 3.1.1 Get all time instants that would incur an early arrival
        early_arrivals = [t for t in range(inst.begin_horizon, inst.earliest_start[task] + 1) if
                               t + max(travel_times_per_bin[inst.bin_per_instant[t]][(inst.depot, task)])
                               <= inst.earliest_start[task]]
        # 3.1.2 If any exist: pick the largest one as the earliest leave time that might be optimal
        if early_arrivals:
            earliest_leaves.append(max(early_arrivals))
        # 3.1.3 Else: do not store anything: earliest leave will then be defined by the segment start
    if earliest_leaves:
        inst.begin_horizon = min(earliest_leaves)
    else:
        inst.begin_horizon = seg_start

    # 3.2 Same for end_horizon: get latest possible extended finish time + worst-case travel time and set it as end_horizon
    latest_returns = [seg_end]
    for task in inst.tasks:
        latest_return = max([t + max(travel_times_per_bin[inst.bin_per_instant[t]][(task, inst.depot)])
                             for t in range(inst.earliest_finish[task], inst.latest_finish_viol[task] + 1)])
        latest_returns.append(latest_return)
    inst.end_horizon = max(latest_returns)

    # 4. Update time bins and mappings between instant and time bins
    inst.bin_per_instant = {t: inst.bin_per_instant[t % inst.last_inst_of_day] for t in range(inst.begin_horizon, inst.end_horizon + 1)}
    inst.instant_per_bin = {int(b): [] for b in inst.bin_per_instant.values()}
    for t in inst.bin_per_instant:
        inst.instant_per_bin[int(inst.bin_per_instant[t])].append(t)
    inst.instant_per_bin = {b: list(sorted(inst.instant_per_bin[b])) for b in inst.instant_per_bin}
    inst.bins = list(inst.instant_per_bin.keys())
    inst.instants = [t for t in range(inst.begin_horizon, inst.end_horizon + 1)]

    # 5. Transfer restricted travel time bins
    add_travel_times_per_bin_restricted(inst, travel_times_per_bin, inst.bins)


    # inst object is manipulated inplace: no need to return it
    return
