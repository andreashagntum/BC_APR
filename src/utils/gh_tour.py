"""This file contains the class GH_tour, which encompasses all information of tours performed by a team.
A separate script is necessary for this to avoid circular imports between branchandprice.py, columngeneration.py and
pricing.py
"""

import math

class GH_tour():
    """Class for describing tours and their relevant information, e.g. worker profile, tasks performed, start and finish
    times of tasks, leave and return time to depot.
    """

    def __init__(self, formation_w_d, formation_id):
        self.formation_w_d = formation_w_d
        self.formation_id = formation_id
        self.tasks = []
        self.quantile_finish_time = {}
        self.worst_case_start_time = {}
        # self.worst_case_finish_time = {}
        self.leave_time = -1
        self.quantile_return_time = -1
        self.worst_case_return_time = -1
        self.tour_cost = math.inf  # sum of weighted expected finish times
        self.tw_viol_prob = {}
        self.is_initial_tour = False  # True iff. tour is one of the tours created at algorithm initialization (i.e. not generated via CG)
        self.is_fake_tour = False

    def __hash__(self):
        """Hash only used to enable usage of GH_tour objects as dict keys. For object comparisons, get_hash(), get_seq_hash(),
        and get_hash_no_comp() can be used.
        """
        return self.get_hash().__hash__()

    def __eq__(self, other):
        """Check if two tours are identical"""
        if self.get_hash() == other.get_hash():
            return True
        else:
            return False

    def copy(self):
        """Copy tour object. Avoids deepcopies whenever possible.
        """
        copy = GH_tour(self.formation_w_d, self.formation_id)
        copy.tasks = self.tasks.copy()
        copy.worst_case_start_time = self.worst_case_start_time.copy()
        copy.quantile_finish_time = self.quantile_finish_time.copy()
        copy.leave_time = self.leave_time
        copy.quantile_return_time = self.quantile_return_time
        copy.worst_case_return_time = self.worst_case_return_time
        copy.tour_cost = self.tour_cost
        copy.cost = self.cost
        copy.tw_viol_prob = self.tw_viol_prob.copy()
        # copy.worst_case_finish_time = self.worst_case_finish_time.copy()
        # copy.start_time_cdf_per_task = self.start_time_cdf_per_task.copy()
        return copy

    def get_hash(self):
        """Compute hash for current tour.
        Hash only depends on the properties of the object, hence can be used to check if two objects of class GH_tour
        are identical.

        Returns
        -------
        str_hash: str
            Customized string-formatted hash that encompasses all information that uniquely defines a tour
        """
        # workaround for fake tour: get trivial fake hash
        if self.is_fake_tour:
            return "faketour"

        # else: compute hash based on tour characteristics
        str_hash = ''
        for skill_level in self.formation_w_d:
            str_hash += str(skill_level) + ":" + str(self.formation_w_d[skill_level]) + ","
        for sl_req in self.skill_comp:
            for sl_act in self.skill_comp[sl_req]:
                str_hash += str(sl_req) + "<-" + str(sl_act) + ":" + str(self.skill_comp[sl_req][sl_act]) + ","
        for task in self.tasks:
            str_hash += str(task) + ":" + str(self.worst_case_start_time[task]) + "," + str(
                self.quantile_finish_time[task]) + ";"
        str_hash += str(self.leave_time) + ","
        str_hash += str(self.quantile_return_time)
        return str_hash

    def get_hash_no_comp(self):
        """Compute hash for current tour, ignoring its skill composition.
        Similar to self.get_hash(), but can be used to check if tours are identical (except for their skill composition).

        Returns
        -------
        str_hash: str
            Customized string-formatted hash that encompasses all information that uniquely defines a tour (excl.
            its skill composition)
        """
        # workaround for fake tour: get trivial fake hash
        if self.is_fake_tour:
            return "faketour"

        # else: compute hash based on tour characteristics
        # generate string hash of tour without considering skill composition
        str_hash = ''
        for skill_level in self.formation_w_d:
            str_hash += str(skill_level) + ":" + str(self.formation_w_d[skill_level]) + ","
        for task in self.tasks:
            str_hash += str(task) + ":" + str(self.worst_case_start_time[task]) + "," + str(
                self.quantile_finish_time[task]) + ";"
        str_hash += str(self.leave_time) + ","
        str_hash += str(self.quantile_return_time)
        return str_hash

    def to_string(self):
        """Convert tour information to printable string.

        Returns
        -------
        str_out: str
            String that describes all properties of the tour in a human-readable format.
        """
        str_out = "Formation-> "
        for skill_level in self.formation_w_d:
            str_out += "level " + str(skill_level) + ":" + str(self.formation_w_d[skill_level]) + " "
        str_out += "\n"
        str_out += "Skill comp.: "
        for sl_req in self.skill_comp:
            if sum(self.skill_comp[sl_req].values()) > 0:
                str_out += str(sl_req) + "<-"
                for sl_act in self.skill_comp[sl_req]:
                    if self.skill_comp[sl_req][sl_act] > 0:
                        str_out += str(sl_act) + ":" + str(self.skill_comp[sl_req][sl_act]) + "x,"
                str_out = str_out.rstrip(",")
                str_out += "; "
        str_out = str_out.rstrip(";")
        str_out += "\n"

        for task in self.tasks:
            str_out += "task" + str(task) + " [" + str(self.worst_case_start_time[task]) + "," + str(
                self.quantile_finish_time[task]) + "[\n"
        str_out += "leave time: " + str(self.leave_time) + ", return time: " + str(self.quantile_return_time) + "\n"
        str_out += "cost: " + str(self.cost) + ", tour cost: " + str(self.tour_cost) + "\n"

        return str_out

    def get_seq_hash(self):
        """Compute hash for current tour that only depends on its executed tasks, their sequence, and the tour's depot
        leave time.

        Returns
        -------
        str_hash: str
            Hash uniquely describing a tour's task sequence and depot leave time.
        """
        str_hash = ''
        for task in self.tasks:
            str_hash += str(task) + ":" + str(self.worst_case_start_time[task]) + "," + str(
                self.quantile_finish_time[task]) + ";"
        str_hash += str(self.leave_time) + ","
        str_hash += str(self.quantile_return_time)
        return str_hash