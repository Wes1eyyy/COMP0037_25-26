#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

import time

from common.scenarios import *
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from generalized_policy_iteration.value_iterator import ValueIterator
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    
    # Get the map for the scenario
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model (p = 0.8 as required)
    airport_environment.set_nominal_direction_probability(0.8)
    
    # =========================================================================
    # Policy Iteration
    # =========================================================================
    print("=" * 70)
    print("  Running Policy Iteration")
    print("=" * 70)

    pi_solver = PolicyIterator(airport_environment)
    pi_solver.initialize()

    start = time.time()
    v_pi, pi_pi = pi_solver.solve_policy()
    pi_time = time.time() - start

    print(f"\nPolicy Iteration finished in {pi_time:.2f}s")

    # =========================================================================
    # Value Iteration
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Running Value Iteration")
    print("=" * 70)

    vi_solver = ValueIterator(airport_environment)
    vi_solver.initialize()

    start = time.time()
    v_vi, pi_vi = vi_solver.solve_policy()
    vi_time = time.time() - start

    print(f"\nValue Iteration finished in {vi_time:.2f}s")

    # =========================================================================
    # Compare policies
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Comparison")
    print("=" * 70)

    # Count how many cells have different actions
    diff_count = 0
    total_cells = 0
    for x in range(airport_map.width()):
        for y in range(airport_map.height()):
            if airport_map.cell(x, y).is_obstruction() or airport_map.cell(x, y).is_terminal():
                continue
            total_cells += 1
            if pi_pi.action(x, y) != pi_vi.action(x, y):
                diff_count += 1

    print(f"  Policy Iteration time:  {pi_time:>8.2f}s")
    print(f"  Value Iteration time:   {vi_time:>8.2f}s")
    print(f"  Speedup (PI / VI):      {pi_time / vi_time:>8.2f}x")
    print(f"  Total non-trivial cells: {total_cells}")
    print(f"  Cells with different policy: {diff_count}")
    print(f"  Policies match: {'Yes' if diff_count == 0 else 'No'}")
    print("=" * 70)
