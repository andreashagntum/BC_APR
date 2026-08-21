"""Functions to monte-carlo sample durations during rolling horizon solving, as well as to apply solutions to the
respective sample: arrival, finish times, and return times are re-computed based on the actual samples values.
"""
import copy
import numpy as np
from config.config import eps_global
from src.instance_loader.instance_loader import add_travel_times_per_bin_restricted
from src.pricing.utils import find_alpha_quantile_pmf

def simulate(inst, inst_seg, th_segment, travel_times_per_bin, rng, all_tours,
             knowledge_state, final_run = False):
    """Simulate travel times, task execution times, and time windows for all tasks executed by a given set of tours.
    For each tour, relevant uncertain parameter are sampled (unless they have been sampled before, in which case they
    are fetched from memory). The tour's attributes are, up to the current segment's start, updated inplace:
    - tour.quantile_finish_time: for all tasks finishing before segment_start, this value then equals the true finish time
                                 of the task under the sampled trajectory
                                 for all tasks finish after segment_start, we compute a distribution as usual
    - tour.quantile_return_time: also update to be in line with the quantile finish time distribution of the last task
    - tour.task_cost_dict: for all tasks finishing before segment_start, this equals the true cost under the sampled trajectory.
                           for all remaining task, this is again based on the respective finish time distribution
    Note: tour attributes are not used downstream, unless this function is called at the very end of a MC solving run.
    In that case, tour attributes are used to print the tour and compute the final (net) solution cost.
    This function also enriches a knowledge_state by whatever information on travel times, service times, and time windows
    is available until the segment's start.


    Parameters
    ---------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    inst_seg: instance_loader.Instance
        inst restricted to the current considered planning horizon
    th_segment: tuple[int, int]
        First and last instant of the considered segment
    travel_times_per_bin: dict
        Maps time bins and edges to their distribution of probabilities
    rng: np.random.RandomState.Generator
        RNG object for sampling (only used if sample is set to True)
    all_tours: list[GH_tour]
        List of conducted tours
    knowledge_state: KnowledgeState
        Object encoding all knowledge over uncertain parameters collected so far, as well as auxiliary variables required
        downstream
    final_run: bool
        If set to True, time windows are set to the sampled values at the very beginning. Should only be used to post-process
        the final results

    Returns
    -------
    all_tours: list[GH_tour]
        List of conducted tours, with their adjusted leave time, return time, quantile finish times, and task costs
    """

    # 0. Preprocessing
    # 0.1 Deepcopy worker usage
    workers = copy.deepcopy(inst.workers)

    # 0.2 Get segment
    seg_start, seg_end = th_segment

    # 0.3 If current simulation run is the final run that is used to compute actual finish times and task costs:
    # Copy sampled time windows to actual time windows, because a correct earliest_finish will be needed for cost calculation
    if final_run:
        inst.earliest_start = {task: inst.sampled_earliest_start[task] for task in inst.earliest_start}
        inst.latest_start = {task: inst.sampled_latest_start[task] for task in inst.latest_start}
        inst.latest_start_viol = {task: inst.sampled_latest_start_viol[task] for task in inst.latest_start_viol}
        inst.latest_finish_viol = {task: inst.sampled_latest_finish_viol[task] for task in inst.latest_finish_viol}
        inst.earliest_finish = {task: inst.sampled_earliest_finish[task] for task in inst.earliest_finish}
        inst.latest_finish = {task: inst.sampled_latest_finish[task] for task in inst.latest_finish}

    # 1. For each tour: sample all relevant parameters
    # Note: time windows and task execution times are once-sampled during instance loading, they are thus only
    # read from inst here
    # 1.0 Sort tours ascending w.r.t. leave time
    all_tours = sorted(all_tours, key = lambda x: x.leave_time)
    # 1.1 Define variables that will be passed down to adjust_inst_to_segment (or used to mutate inst_seg inplace)
    # We track three changes that need to be implemented in the inst object:
    # - distributions for all time bins and edges which are traversed during seg_start (all probability mass before seg_start will be added to seg_start)
    # - task execution times for tasks that are performed during seg_start, and haven't finished even though their finish
    #   time (according to inst_seg) was before seg_start (here, we use start_time - seg_start + 1 as new task execution time)
    # - (earliest) start times for tasks where the team is already at the location and the earliest start was before
    #   seg_start, but handling has not started yet: in this case, the actual earliest start is definitely later (here, we use seg_start + 1 as new earliest start time)
    # 1.2 Iteratively assign workers to tour and compute their finish times
    for tour in all_tours:
        always_feas_edges = []  # list of all edges for which feasibility checks will be bypassed during BPC&S
        finished_active_tasks = []  # list of all tasks in current tour that have already been finished
        is_active = False # True iff. tour is active at seg_start
        # 2. Only consider tours that were not sampled yet and that are scheduled to start before the segment start
        if tour.leave_time < seg_start:
            # 2.1 Initialize loop: current nodes is depot, leave time is earliest point of time where sufficient workforce is available
            leave_time = tour.leave_time  # earliest point in time where enough workers of all skill levels are available
            for k in tour.skill_comp_cnt:
                leave_time = max(leave_time, min([t for t in workers[k] if t >= leave_time and workers[k][t] >= tour.skill_comp_cnt[k]]))
            tour.leave_time = leave_time
            curr = "depot"
            curr_time = leave_time

            # 2.3 Iteratively compute finish times for task edges
            for succ in tour.tasks + ["depot"]:
                curr_time_bin = inst.bin_per_instant[curr_time]
                # Encode nodes into the name structure (source/sink instead of depot) that PricingNetwork expects
                curr_node = "source" if curr == "depot" else curr
                succ_node = "sink" if succ == "depot" else succ
                # 2.3.1 Sample random travel time if needed
                if (curr_node, succ_node) not in inst.sampled_tts[curr_time_bin]:
                    items = travel_times_per_bin[curr_time_bin][(curr, succ)].items()
                    # Sample and store travel times
                    inst.sampled_tts[curr_time_bin][(curr_node, succ_node)] = rng.choice([k for (k, _) in items], p = [v for (_, v) in items])
                    inst_seg.sampled_tts[curr_time_bin][(curr_node, succ_node)] = inst.sampled_tts[curr_time_bin][(curr_node, succ_node)]
                # 2.3.2 Compute start and finish time
                # If succ is task: travel + task execution
                if succ != "depot":
                    arrival_time = curr_time + inst.sampled_tts[curr_time_bin][(curr_node, succ_node)]
                    start_time = max(inst.sampled_earliest_start[succ], arrival_time)
                    finish_time = start_time + inst.sampled_modes_with_domination[succ][tour.formation_id]
                    tour.quantile_finish_time[succ_node] = finish_time
                # Else: just travel
                else:
                    arrival_time = curr_time + inst.sampled_tts[curr_time_bin][curr_node, succ_node]
                    start_time = arrival_time
                    finish_time = start_time
                    tour.quantile_finish_time[succ_node] = finish_time
                #  Remember edge in list of always-feasible edges
                always_feas_edges.append((curr_node, succ_node, curr_time_bin))
                # 2.3.4 If finish time lies before seg_start: update finish time, return time, and task cost
                if finish_time <= seg_start:
                    finished_active_tasks.append(succ)
                    if succ != "depot":
                        # Fixed cost
                        tour.task_cost_dict[succ] = inst.weights[succ] * finish_time
                        # Quadratic penalty for delays
                        if finish_time > inst.sampled_latest_finish[succ]:
                            tour.task_cost_dict[succ] += inst.weights[succ] * (finish_time - inst.sampled_latest_finish[succ]) ** 2
                        # Store observed values: time window, task execution time, AND travel time
                        # 1. Travel time
                        knowledge_state.store_travel_time_knowledge(curr_node, succ_node, curr_time_bin,
                                                                    {inst.sampled_tts[curr_time_bin][(curr_node, succ_node)]: 1})
                        # 2. Time window
                        knowledge_state.store_time_window_knowledge(inst, succ)
                        # 3. Task execution time
                        knowledge_state.store_service_time_knowledge(succ_node, tour.formation_id,
                                                                     inst.sampled_modes_with_domination[succ][tour.formation_id])
                    else: # successor is the depot: need to update quantile return time only
                        tour.quantile_return_time = finish_time
                        # Store observed values:
                        # 1. Only time window
                        knowledge_state.store_travel_time_knowledge(curr_node, succ_node, curr_time_bin,
                                                                    {inst.sampled_tts[curr_time_bin][(curr_node, succ_node)]: 1})

                # 2.3.5 Else: Update travel times, service times, and time windows depending on what part of it has
                # already been observed
                else:
                    # 2.3.5.1 Store currently active task, all finished tasks, and all upcoming (aka passive) tasks
                    tour.curr_active_task = succ
                    tour.finished_active_tasks = finished_active_tasks
                    tour.active_tasks = finished_active_tasks + [tour.curr_active_task]
                    tour.passive_tasks = [task for task in tour.tasks if (task != tour.curr_active_task and
                                                                          task not in tour.finished_active_tasks)]
                    knowledge_state.always_feas_edges += always_feas_edges
                    # I. Travel times:
                    # If arrival time is before or at segment start: TT fully known
                    if arrival_time <= seg_start:
                        knowledge_state.store_travel_time_knowledge(curr, succ, curr_time_bin,
                                                                    {inst.sampled_tts[curr_time_bin][(curr_node, succ_node)]: 1})
                    # Else: Can rule out TT events that would lead to an arrival before seg_start
                    else:
                        new_distr = compute_updated_distribution(curr_time, seg_start, (curr, succ), travel_times_per_bin[curr_time_bin])
                        knowledge_state.store_travel_time_knowledge(curr, succ, curr_time_bin, new_distr)

                    # Next: time windows and service times => only if successor node is not the depot
                    if succ == "depot":
                        knowledge_state.tours_returning_to_depot.append(tour)
                        break

                    else:
                        # II. Time Windows:
                        # II.1 team arrived at the task AND has already started it: can update time window to the sampled values
                        if arrival_time <= seg_start and start_time <= seg_start:
                            # II.1.1 If task is already finished: entire time window, incl. latest start is known
                            # This case can not occur, so we skip it

                            # II.1.2 Else: only ES, LF, and LF_v are known: EF, LS, LS_v all depend on the actual task
                            # execution time and can thus not be estimated more precisely
                            # In this case, we simply offset all time window values by the difference between our ES estimate
                            # and the sampled ES value
                            offset = inst.sampled_earliest_start[succ] - inst.earliest_start[succ]
                            inst.earliest_start[succ] += offset
                            inst.latest_start[succ] += offset
                            inst.latest_start_viol[succ] += offset
                            inst.latest_finish_viol[succ] += offset
                            inst.earliest_finish[succ] += offset
                            inst.latest_finish[succ] += offset

                        # II.2 If team arrived at the task AND task has NOT yet been started:
                        # Update ES to seg_start + 1 IF estimated ES was before or at seg_start
                        if arrival_time <= seg_start and start_time > seg_start:
                            if inst.earliest_start[succ] <= seg_start:
                                offset = seg_start + 1 - inst.earliest_start[succ]
                                inst.earliest_start[succ] += offset
                                inst.latest_start[succ] += offset
                                inst.latest_start_viol[succ] += offset
                                inst.latest_finish_viol[succ] += offset
                                inst.earliest_finish[succ] += offset
                                inst.latest_finish[succ] += offset

                        # III. Service times
                        # III.1 If task is finished: service time is known
                        # This case can not occur, so we skip it

                        # III.2 If task is ongoing, not yet finished, but *should* have been finished if estimated service time
                        # was correct: update service time
                        if start_time <= seg_start and finish_time > seg_start:
                            if start_time + inst.modes_with_domination[succ][tour.formation_id] <= seg_start:
                                knowledge_state.store_service_time_knowledge(succ, tour.formation_id, seg_start - start_time + 1)


                    # 2.3.5 Remember tour as active
                    knowledge_state.tours_from_prev_segs.append(tour)


                    # 2.3.6 Also remember its finished active tasks and its currently active task
                    knowledge_state.finished_active_tasks += finished_active_tasks
                    knowledge_state.active_tasks.append(succ)
                    knowledge_state.passive_tasks += [tsk for tsk in tour.tasks if (tsk != succ) and (tsk not in finished_active_tasks)]


                    # break out of loop: we have now simulated the last task that matters for downstream computations
                    # NOTE: If the tour objects from all_tours should be passed to the initial column pool, this
                    # is NOT possible and one instead needs to continue with exact convolutions from here on
                    # before breaking, we set all remaining values to dummy values to ensure these are not used anywhere downstream
                    break


                # 2.3.5 Update curr_time and curr node
                curr_time = finish_time
                curr = succ


        # 3. Occupy workers
        for t in range(tour.leave_time, tour.quantile_return_time):
            for k in tour.skill_comp_cnt:
                workers[k][t] -= tour.skill_comp_cnt[k]


    # 4. Advance knowledge time
    knowledge_state.t = seg_start


    return all_tours



def adjust_inst_to_segment(inst, th_segment, travel_times_per_bin, lookahead, all_tours,
                           knowledge_state):
    """Adjust a (deepcopied version of the input) inst object according to a passed th_segment.
    All relevant datastructures, such as task lists, feasible modes, tasks per mode, and time horizons are adjusted so
    they only refer to tasks that are part of the current th_segment. This includes tours from previous segments that
    are still active at the start of the current segment.

    Parameters
    ---------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    th_segment: tuple[int, int]
        First and last instant of the considered segment
    travel_times_per_bin: dict
        Maps time bins and edges to their distribution of probabilities
    lookahead: int
        Length of interval whose tasks will be considered during planning of each segment. Tasks that start after the segment
        end will be planned, but their solution will not be implemented and might be overwritten by the decision in the
        subsequent segment.
    all_tours: list[GH_tour]
        List of all tours (finished or unfinished) from previous segments
    knowledge_state: KnowledgeState
        Object encoding all knowledge over uncertain parameters collected so far, as well as auxiliary variables required
        downstream

    Returns
    -------
    all_tours: list[GH_tour]
        List of all tours from previous segments that finished before seg_start
    """


    seg_start, seg_end = th_segment

    # 1. Get all tours of prior segments, that are still active at the beginning of the considered segment
    # These tours are also called 'active tours'
    inst.tours_from_prev_segs = knowledge_state.tours_from_prev_segs

    # 2. Get some task lists
    # 2.1 All tasks with their earliest start in this segment
    # Note: this might contain tasks that were scheduled in the last segment
    tasks_in_segment = [task for task in inst.tasks if (inst.earliest_start[task] >= seg_start and
                                                        inst.earliest_start[task] < seg_start + lookahead)]
    # 2.2 All tasks that are part of an active tour and have been finished
    finished_active_tasks = knowledge_state.finished_active_tasks
    # 2.3 All tasks that are part of an active task and are currently being executed/traveled to
    # Note: this list might contain tasks that are also part of tasks_in_segment!
    active_tasks = knowledge_state.active_tasks
    # 2.4 All tasks that are part of an active tour, but have not been executed
    # Note: This list also contains task that were scheduled for execution in the last segment, but were not executed
    # yet due to workforce shortage or other delay
    passive_tasks = knowledge_state.passive_tasks
    # 2.4 Superset of all tasks
    all_tasks = list(set(tasks_in_segment + finished_active_tasks + active_tasks + passive_tasks))
    # 2.5 All upcoming tasks: these are the tasks that we can freely assign to any tour (and, by that, also freely
    # choose their formation
    upcoming_tasks = list(set(all_tasks).difference((set(active_tasks).union(finished_active_tasks))))

    # Store tasks in inst objectr
    inst.tasks = all_tasks
    inst.tasks_from_prev_segs = active_tasks + finished_active_tasks

    # 3. Store observed edges, i.e., edges for which travel times are known and which can always be traversed without
    # violating chance constraints or time windows
    # Note: this is necessary because said edges need to be used by the BPC&S algorithm in order to construct the part
    # of active tours that lies before seg_start. During simulate(), it might be that tours are postponed so much
    # that they actually violate chance constraints or extended time windows. the latter two constraints will NOT
    # be checked by the BPC&S for always_feas_edges.
    inst.always_feas_edges = knowledge_state.always_feas_edges

    # 3. Update all_tours to only contain tours that will not be changed anymore
    all_tours = [tour for tour in all_tours if (tour not in inst.tours_from_prev_segs and tour.leave_time < seg_start)]

    # 4. Restrict all sets relating to tasks
    # 4.1 Time windows
    inst.earliest_start = {task: inst.earliest_start[task] for task in all_tasks}
    inst.latest_start = {task: inst.latest_start[task] for task in all_tasks}
    inst.earliest_finish = {task: inst.earliest_finish[task] for task in all_tasks}
    inst.latest_finish = {task: inst.latest_finish[task] for task in all_tasks}
    inst.latest_start_viol = {task: inst.latest_start_viol[task] for task in all_tasks}
    inst.latest_finish_viol = {task: inst.latest_finish_viol[task] for task in all_tasks}


    # 4.2 Modes, weights, task locations, and tasks per formation
    inst.modes = {task: inst.modes[task] for task in all_tasks}
    inst.modes_with_domination = {task: inst.modes_with_domination[task] for task in all_tasks}
    inst.weights = {task: inst.weights[task] for task in all_tasks}
    inst.task_locations = {task: inst.task_locations[task] for task in all_tasks}
    # 4.2.1 For tasks of active tours that have already been executed or are currently being executed: only copy single
    # formation
    for tour in inst.tours_from_prev_segs:
        for task in tour.finished_active_tasks + [tour.curr_active_task]:
            assert tour.formation_id in inst.modes[task] or tour.formation_id in inst.modes_with_domination[task]  # quick sanity check
            # if formation_id is dominated for task: modes list is empty
            if tour.formation_id in inst.modes[task]:
                inst.modes[task] = {tour.formation_id: inst.modes[task][tour.formation_id]}
            else:
                inst.modes[task] = {}
            # formation_id will always be in modes_with_domination
            inst.modes_with_domination[task] = {tour.formation_id: inst.modes_with_domination[task][tour.formation_id]}

    # 4.3 Tasks per formation: formations with no tasks will be dropped
    inst.tasks_per_formation = {f: list(set(inst.tasks_per_formation[f]).intersection(set(upcoming_tasks)))
                                for f in inst.tasks_per_formation}
    # 4.3.1 Add tasks from previous segment separately, because inst.tasks_per_formation still contains all formations for those
    for tour in inst.tours_from_prev_segs:
        for task in tour.tasks:
            # only add task if mode is not dominated for it
            if task in active_tasks + finished_active_tasks and tour.formation_id in inst.modes[task]:
                inst.tasks_per_formation[tour.formation_id].append(task)
    # 4.3.2 Drop formations with no suitable tasks
    to_drop = [f for f in inst.tasks_per_formation if not inst.tasks_per_formation]
    for f in to_drop:
        del inst.tasks_per_formation[f]

    # 4.4 Same for tasks per formation with domination
    inst.tasks_per_formation_with_domination = {f: list(set(inst.tasks_per_formation_with_domination[f]).intersection(set(upcoming_tasks)))
                                                for f in inst.tasks_per_formation_with_domination}
    for tour in inst.tours_from_prev_segs:
        for task in tour.tasks:
            if task in active_tasks + finished_active_tasks:
                inst.tasks_per_formation_with_domination[tour.formation_id].append(task)
    to_drop_with_domination = [f for f in inst.tasks_per_formation_with_domination if not inst.tasks_per_formation_with_domination]
    for f in to_drop_with_domination:
        del inst.tasks_per_formation_with_domination[f]

    # 4.5 Drop formations entry for all formations that are not suitable for any task, neither normally nor with domination
    for f in set(to_drop).intersection(set(to_drop_with_domination)):
        del inst.formations[f]
        del inst.formations_w_d[f]


    # 5. Restrict time horizon
    # 5.1 Get earliest possible leave time for any task: this will be the horizon begin
    # Initialize list of earliest leaves with leave time of earliest tours from previous segment that is still active
    # (Note: this is necessary to ensure all travel times values accessed during pricing actually exist)
    earliest_leaves = [tour.leave_time for tour in inst.tours_from_prev_segs]
    for task in all_tasks:
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
    for task in all_tasks:
        latest_return = max([t + max(travel_times_per_bin[inst.bin_per_instant[t % inst.last_inst_of_day]][(task, inst.depot)])
                             for t in range(inst.earliest_finish[task], inst.latest_finish_viol[task] + 1)])
        latest_returns.append(latest_return)
    inst.end_horizon = max(latest_returns)

    # 6. Update time bins and mappings between instant and time bins
    inst.bin_per_instant = {t: inst.bin_per_instant[t % inst.last_inst_of_day] for t in range(inst.begin_horizon, inst.end_horizon + 1)}
    inst.instant_per_bin = {int(b): [] for b in inst.bin_per_instant.values()}
    for t in inst.bin_per_instant:
        inst.instant_per_bin[int(inst.bin_per_instant[t])].append(t)
    inst.instant_per_bin = {b: list(sorted(inst.instant_per_bin[b])) for b in inst.instant_per_bin}
    inst.bins = list(inst.instant_per_bin.keys())
    inst.instants = [t for t in range(inst.begin_horizon, inst.end_horizon + 1)]

    # 8. Transfer restricted travel time bins
    add_travel_times_per_bin_restricted(inst, travel_times_per_bin, inst.bins)

    # 6. Store task list
    inst.tasks = all_tasks

    return all_tours

def restrict_workforce(inst, inst_seg, knowledge_state, travel_times_per_bin,
                       th_segments, seg_idx, all_tours):
    """Restrict workforce to considered planning horizon
    Note: we no longer adjust the workforce according to the solutions from the previous segments because
    workers occupied by tours from the previous segments that are still active during the new segment's time horizon
    will be force-occupied downstream anyway: for each existing tour, an artificial label in the pricing network
    will be created with a matching leave time, formation, and task sequence (up until the last task that was
    finished before the time horizon). This tour then occupies the exact number of workers that it also occupied
    in the solution of the last segment.
    There are two exceptions for this:
    A) tours that are returning to the depot and have not returned yet at the segment start:
    We need to adjust the available workforce according to their occupancy
    B) tours that have already returned to the depot between [inst_seg.begin_horizon, seg_start]:+
    For these tours, we do not need to compute quantile return times because we have already observed them.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    inst_seg: instance_loader.Instance
        instance_loader.Instance object restrict to the current segment
    knowledge_state: KnowledgeState
        Object encoding all knowledge over uncertain parameters collected so far, as well as auxiliary variables required
        downstream
    travel_times_per_bin: dict
        Maps time bins and edges to their distribution of probabilities
    th_segments: list[(int, int)]
       Tuples indicating the start and end of a segment (both time instants included)
    seg_idx: int
       Index of current segment in th_segments
    all_tours: list[GH_tour]
        List of conducted tours
    """

    inst_seg.workers = copy.deepcopy(inst.workers)
    inst_seg.workers = {k: {t: v for (t, v) in inst_seg.workers[k].items() if
                            (t >= inst_seg.begin_horizon and t <= inst_seg.end_horizon)}
                        for k in inst_seg.workers}
    # Case A)
    for tour in knowledge_state.tours_returning_to_depot:
        for k in tour.skill_comp_cnt:
            # Compute quantile based on quantile finish time of last task + distribution between task and sink
            # Note: the latter is ALWAYS given by new_distrs
            qft_at_last_task = sorted(tour.quantile_finish_time.values(), reverse=True)[1]
            time_bin = inst.bin_per_instant[qft_at_last_task]
            tour.quantile_return_time = qft_at_last_task + find_alpha_quantile_pmf(
                travel_times_per_bin[time_bin][tour.tasks[-1], "depot"],
                inst.worker_quantile)
            assert tour.quantile_return_time > th_segments[seg_idx][0]
            for t in range(tour.leave_time, tour.quantile_return_time):
                if t in inst_seg.workers[k]:
                    inst_seg.workers[k][t] -= tour.skill_comp_cnt[k]
                    for kk in [kk for kk in inst_seg.workers_w_d if kk <= k]:
                        inst_seg.workers_w_d[kk][t] -= tour.skill_comp_cnt[k]
    # Case B)
    for tour in all_tours:
        if tour.quantile_return_time >= inst_seg.begin_horizon and tour.quantile_return_time <= th_segments[seg_idx][0]:
            for k in tour.skill_comp_cnt:
                for t in range(tour.leave_time, tour.quantile_return_time):
                    if t in inst_seg.workers[k]:
                        inst_seg.workers[k][t] -= tour.skill_comp_cnt[k]
                        for kk in [kk for kk in inst_seg.workers_w_d if kk <= k]:
                            inst_seg.workers_w_d[kk][t] -= tour.skill_comp_cnt[k]
    inst_seg.workers_w_d = {k: {t: v for (t, v) in inst_seg.workers_w_d[k].items() if
                                (t >= inst_seg.begin_horizon and t <= inst_seg.end_horizon)}
                            for k in inst_seg.workers_w_d}

    return


def compute_updated_distribution(curr_time, seg_start, e, distr):
    """Update a travel time distribution. Assumes that a team started traversing an arc e starting at time 'curr_time',
    and has not arrived at its destination at time 'seg_start'.

    Parameters
    ----------
    curr_time: int
       Time instant at which traversing arc e has begun
    seg_start: int
       Segment start (or, in other words, the time instant at which the system is evaluated/observed)
    e: tuple(str, str)
       Edge that is being traversed
    distr: dict
       PMF of edge e for time bin at time curr_time

    Returns
    -------
    distr: dict
       Updated PMF
    """

    # Derive distribution by cutting the actual distribution at seg_start
    new_distr = {}
    for t in sorted(distr[e]):
        if curr_time + t <= seg_start:
            if seg_start - curr_time not in new_distr:
                new_distr[seg_start - curr_time] = 0
            new_distr[seg_start - curr_time] += distr[e][t]
        else:
            new_distr[t] = distr[e][t]
    # Re-sort travel time dict for safety and store in knowledge base
    new_distr = dict(sorted(new_distr.items(), key=lambda x: x[0]))

    return new_distr

class KnowledgeState:
    """This class encodes the knowledge state of the simulator at a given time instant. It can be used to summarize
    either (a) all parameters that are known to the DM or (b) all parameters that are unknown to the DM (i.e., still
    subject to uncertainty).

    This object acts twofold. First, it's an up-to-date version of the instance - it contains all values that have realized
    already (e.g. time windows that have passed, task execution times of tasks that have been finished), as well as updated
    estimates on parameters that have been partially observed (e.g. updated time window if the earliest start of a task
    was in the past, but the team was still not able to start the task: an updated TW could then be current_instant+1).
    Second, it contains several attributes that are accessed by adjust_inst_to_segment to adjust the InstanceLoader
    object to the current segment.

    """

    def __init__(self, t):
        # Initialize all properties of the InstanceLoader instance that are subject to uncertainty and realize over time
        # 1. Current time instant
        self.instant = t

        # 2. Time-window related attributes
        self.earliest_start = {}
        self.latest_start = {}
        self.latest_start_viol = {}
        self.latest_finish_viol = {}
        self.earliest_finish = {}
        self.latest_finish = {}

        # 3. Task execution-time related attributes
        self.modes = {}
        self.modes_with_domination = {}

        # 4. Time windows
        self.travel_times_per_bin = {}

        # 4. Other attributes that are used for segment adjustment and instance solving
        self.tours_from_prev_segs = [] # tours that are executing or traveling to a task at seg_start
        self.tours_returning_to_depot = [] # tours that are returning to the depot during seg_start
        self.active_tasks = [] # tasks of tours active at seg_start that are currently being executed/traveled to
        self.passive_tasks = [] # tasks of tours active at seg_start that are not yet executed/traveled to
        self.finished_active_tasks = [] # tasks of tours active at seg_start that have already finished
        self.always_feas_edges = [] # list of edges whose travel time was already observed



    def store_time_window_knowledge(self, inst, task):
        """Store FULL knowledge of the time window for a given task.
        If time windows are only partially known (e.g. earliest start is known, but latest start is not), then this function
        should NOT be used.
        """
        self.earliest_start[task] = inst.sampled_earliest_start[task]
        self.latest_start[task] = inst.sampled_latest_start[task]
        self.latest_start_viol[task] = inst.sampled_latest_start_viol[task]
        self.latest_finish_viol[task] = inst.sampled_latest_finish_viol[task]
        self.earliest_finish[task] = inst.sampled_earliest_finish[task]
        self.latest_finish[task] = inst.sampled_latest_finish[task]


    def store_travel_time_knowledge(self, pred, succ, time_bin, tt):
        """Store knowledge on the travel time between two nodes in a given time bin.
        """
        if time_bin not in self.travel_times_per_bin:
            self.travel_times_per_bin[time_bin] = {}
        self.travel_times_per_bin[time_bin][pred, succ] = tt


    def store_service_time_knowledge(self, task, formation_id,
                                     service_time):
        """Store knowledge on the task execution time of a given task by a given formation.
        """
        if task not in self.modes_with_domination:
            self.modes_with_domination[task] = {}
        self.modes_with_domination[task][formation_id] = service_time

        if task not in self.modes:
            self.modes[task] = {}
        if formation_id in self.modes[task]:
            self.modes[task][formation_id] = service_time

    def join(self, inst, travel_times_per_bin):
        """Joins all information stored in the knowledge state into the instance.
        """

        # 1. Time windows
        for attr in ["earliest_start", "latest_start", "latest_start_viol", "earliest_finish", "latest_finish",
                     "latest_finish_viol"]:
            getattr(inst, attr).update(getattr(self, attr))

        # 2. Task execution times
        for task in self.modes:
            inst.modes[task].update(self.modes[task])
        for task in self.modes_with_domination:
            inst.modes_with_domination[task].update(self.modes_with_domination[task])

        # 3. Travel times
        for time_bin in self.travel_times_per_bin:
            travel_times_per_bin[time_bin].update(self.travel_times_per_bin[time_bin])

        return travel_times_per_bin

    def reset(self):
        """Reset the knowledge state as preparation for the next segment.
        """
        self.tours_from_prev_segs = []
        self.tours_returning_to_depot = []
        self.active_tasks = []
        self.active_tasks = []
        self.finished_active_tasks = []
        self.always_feas_edges = []
