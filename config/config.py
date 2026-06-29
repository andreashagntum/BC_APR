"""This file defines some threshold parameters used throughout the algorithm.
It can also be extended by additional parameters that should be used by other functions, independent of the specific
instance that is to be solved.
"""

eps_global = 10**(-4) # global epsilon used if no other value is specified
eps_col_neg = - 10 ** (-4) # maximum cost for a column considered to be negative
eps_gc_round = 5 * 10** (-4) # gomory cut rounding triggered once coefficient of a sublabel falls below this value
eps_gc_recalc = 10 ** (-3) # gomory cut recalculation triggered once coefficient of a sublabel falls below this value
eps_gap = 10**(-6) # maximum gap for a solution considered to be optimal for the integer problem
elapsed_time_buffer_factor = 1.05 # factor that allows the last CG step to slightly exceed the maximum BPC&S runtime (to allow for finishing solving the last evaluated tree node)
alpha_tol = 10**(-4) # summand that allows slight violation of alpha chance constraints (to avoid errors due to numerical imprecision)