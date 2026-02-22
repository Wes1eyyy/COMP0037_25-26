#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

import time
import numpy as np

from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

def compute_value_stats(v, airport_map):
    """Compute summary statistics of the converged value function."""
    vals = []
    for x in range(airport_map.width()):
        for y in range(airport_map.height()):
            cell = airport_map.cell(x, y)
            if cell.is_obstruction() or cell.is_terminal():
                continue
            val = v.value(x, y)
            if not np.isnan(val):
                vals.append(val)
    vals = np.array(vals)
    return vals.mean(), vals.min(), vals.max(), vals.std()

if __name__ == '__main__':

    # Get the map for the scenario
    airport_map, drawer_height = full_scenario()

    results = []

    for p in [1, 0.9, 0.6, 0.3]:

        print(f"\n{'='*60}")
        print(f"Running Policy Iteration with p = {p}")
        print(f"{'='*60}")

        # Set up the environment
        airport_environment = LowLevelEnvironment(airport_map)
        airport_environment.set_nominal_direction_probability(p)

        # Create the policy iterator
        policy_solver = PolicyIterator(airport_environment)
        policy_solver.set_gamma(0.99)
        policy_solver.initialize()

        # Bind drawers
        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)

        value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_function_drawer)

        # Solve with timing
        t_start = time.time()
        v, pi = policy_solver.solve_policy()
        t_elapsed = time.time() - t_start

        # Compute value function statistics
        v_mean, v_min, v_max, v_std = compute_value_stats(v, airport_map)

        # Save screenshots automatically
        p_str = str(p).replace('.', '_')
        policy_drawer.save_screenshot(f"q3d_policy_p{p_str}.pdf")
        value_function_drawer.save_screenshot(f"q3d_value_p{p_str}.pdf")

        results.append({
            'p': p,
            'time': t_elapsed,
            'v_mean': v_mean,
            'v_min': v_min,
            'v_max': v_max,
            'v_std': v_std,
        })

        print(f"p={p}: time={t_elapsed:.2f}s, V_mean={v_mean:.1f}, V_min={v_min:.1f}, V_max={v_max:.1f}, V_std={v_std:.1f}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'SUMMARY TABLE':^80}")
    print(f"{'='*80}")
    print(f"{'p':>6} | {'Time (s)':>10} | {'V_mean':>10} | {'V_min':>10} | {'V_max':>10} | {'V_std':>10}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for r in results:
        print(f"{r['p']:>6.1f} | {r['time']:>10.2f} | {r['v_mean']:>10.1f} | {r['v_min']:>10.1f} | {r['v_max']:>10.1f} | {r['v_std']:>10.1f}")
    print(f"{'='*80}")

    print("\nAll done. Close the window or press any key.")
