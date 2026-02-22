#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

import time

from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    
    # Get the map for the scenario
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Q3e: p = 0.8 for the rest of the coursework
    airport_environment.set_nominal_direction_probability(0.8)

    # Parameter configurations to test: (theta, max_eval_steps)
    configs = [
        (1e-6, 100),   # baseline (most precise)
        (1e-6, 50),
        (1e-6, 20),
        (1e-6, 10),
        (1e-6, 5),
        (1e-4, 100),
        (1e-4, 20),
        (1e-4, 10),
        (1e-2, 100),
        (1e-2, 20),
        (1e-2, 10),
        (1,    100),
        (1,    20),
    ]

    # Store results
    results = []

    # Run the baseline first to get a reference policy
    baseline_solver = PolicyIterator(airport_environment)
    baseline_solver.set_theta(1e-6)
    baseline_solver.set_max_policy_evaluation_steps_per_iteration(100)
    baseline_solver.initialize()

    start = time.time()
    v_base, pi_base = baseline_solver.solve_policy()
    baseline_time = time.time() - start

    print("=" * 80)
    print(f"  Baseline (theta=1e-6, max_steps=100): {baseline_time:.2f}s")
    print("=" * 80)

    for theta, max_steps in configs:
        # Create a fresh solver for each configuration
        solver = PolicyIterator(airport_environment)
        solver.set_theta(theta)
        solver.set_max_policy_evaluation_steps_per_iteration(max_steps)
        solver.initialize()

        # Time the solve
        start = time.time()
        v, pi = solver.solve_policy()
        elapsed = time.time() - start

        # Check if the resulting policy matches the baseline
        policy_match = True
        for x in range(airport_map.width()):
            for y in range(airport_map.height()):
                if airport_map.cell(x, y).is_obstruction() or airport_map.cell(x, y).is_terminal():
                    continue
                if pi.action(x, y) != pi_base.action(x, y):
                    policy_match = False
                    break
            if not policy_match:
                break

        results.append((theta, max_steps, elapsed, policy_match))
        print(f"theta={theta:<8g}  max_steps={max_steps:<4d}  "
              f"time={elapsed:>7.2f}s  policy_match={policy_match}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"  {'theta':>10}  {'max_steps':>10}  {'time (s)':>10}  {'policy OK':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for theta, max_steps, elapsed, match in results:
        print(f"  {theta:>10.1e}  {max_steps:>10d}  {elapsed:>10.2f}  {'Yes' if match else 'NO':>10}")
    print("=" * 80)
