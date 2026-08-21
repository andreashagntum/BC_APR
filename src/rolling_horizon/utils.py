"""Utility functions used for solving instances with a rolling horizon approach.
"""
import matplotlib.pyplot as plt
from src.instance_loader.instance_loader import add_travel_times_per_bin_restricted

def min_val(d):
    return min(d.keys())

def max_val(d):
    return max(d.keys())

def expected_val(d):
    mean_rounded = int(sum([k * v for (k,v) in d.items()]))
    # get closest key ind
    closest = min(d.keys(), key=lambda v: abs(v - mean_rounded))
    return closest



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




def plot_workforce_usage(inst, all_tours):
    """Create and show a plot of the worker usage over time.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files
    all_tours: list[GH_tour]
        List of conducted tours, with their adjusted leave time, return time, quantile finish times, and task costs
    """

    # 1. Compute workforce usage
    avail_workers = {k: inst.workers[k][inst.begin_horizon] for k in inst.workers}
    workers_used = {}
    for tour in all_tours:
        for k in tour.skill_comp_cnt:
            if k not in workers_used:
                workers_used[k] = {}
            for t in range(tour.leave_time, tour.quantile_return_time):
                if t not in workers_used[k]:
                    workers_used[k][t] = 0
                workers_used[k][t] += tour.skill_comp_cnt[k]

    # 2. Create plot
    levels = sorted(workers_used.keys())
    n = len(levels)

    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, level in zip(axes, levels):
        usage_by_time = workers_used[level]
        times = sorted(usage_by_time.keys())
        usage = [usage_by_time[t] for t in times]
        ax.plot(times, usage, marker="o", label="Usage")
        if level in avail_workers:
            capacity = avail_workers[level]
            ax.axhline(capacity, color="red", linestyle="--", label="Available capacity")
            ax.set_ylim(top=max(capacity, max(usage, default=0)) * 1.1)
        ax.set_title(f"Worker level {level}")
        ax.set_ylabel("Resource usage")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    return