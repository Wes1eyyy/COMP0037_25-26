#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment

def run_policy_evaluation(airport_map, drawer_height, gamma, tag, num_iters=1000):
    """Run policy evaluation with the given gamma and save a PDF screenshot."""
    airport_environment = LowLevelEnvironment(airport_map)
    airport_environment.set_nominal_direction_probability(1)

    policy_solver = PolicyIterator(airport_environment)
    policy_solver.set_gamma(gamma)
    policy_solver.initialize()
    policy_solver.set_max_policy_evaluation_steps_per_iteration(10)

    V = policy_solver.evaluate_policy()
    value_function_drawer = ValueFunctionDrawer(V, drawer_height)

    for steps in range(num_iters):
        policy_solver.evaluate_policy()
        value_function_drawer.update()

    value_function_drawer.save_screenshot(f"q3b_value_{tag}.pdf")
    print(f"Saved q3b_value_{tag}.pdf")
    return value_function_drawer

if __name__ == '__main__':

    airport_map, drawer_height = full_scenario()

    # gamma = 1: non-convergent (50 iters is enough to show divergence)
    run_policy_evaluation(airport_map, drawer_height, gamma=1.0, tag="gamma1", num_iters=50)

    # gamma = 0.99: convergent
    drawer = run_policy_evaluation(airport_map, drawer_height, gamma=0.99, tag="gamma099")

    drawer.wait_for_key_press()
