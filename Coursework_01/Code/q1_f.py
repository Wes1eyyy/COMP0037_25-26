#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import full_scenario
from p1.high_level_environment import PlannerType
from q1_e import print_results, run_planner_on_all_bins

if __name__ == '__main__':

    col = 56

    # --- Run 1: Dijkstra WITHOUT traversability costs (Euclidean only) ---
    airport_map, drawer_height = full_scenario()
    airport_map.set_use_cell_type_traversability_costs(False)

    results_no_alpha = run_planner_on_all_bins(airport_map, PlannerType.DIJKSTRA,
                                               show_graphics=False)
    cost_no_alpha, cells_no_alpha = print_results('Dijkstra (no traversability costs)',
                                                   results_no_alpha)

    # --- Run 2: Dijkstra WITH traversability costs (alpha-weighted) ---
    airport_map, drawer_height = full_scenario()
    airport_map.set_use_cell_type_traversability_costs(True)

    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    airport_map_drawer.wait_for_key_press()

    results_alpha = run_planner_on_all_bins(airport_map, PlannerType.DIJKSTRA,
                                            show_graphics=True)
    cost_alpha, cells_alpha = print_results('Dijkstra (with traversability costs)',
                                            results_alpha)

    # --- Head-to-head summary ---
    print(f"\n{'=' * col}")
    print(f"  Comparison: effect of traversability costs on Dijkstra")
    print(f"{'=' * col}")
    print(f"  {'Mode':>34}  {'Total cost':>10}  {'Cells visited':>13}")
    print(f"  {'-'*34}  {'-'*10}  {'-'*13}")
    print(f"  {'Without traversability costs':>34}  {cost_no_alpha:>10.2f}  {cells_no_alpha:>13}")
    print(f"  {'With traversability costs':>34}  {cost_alpha:>10.2f}  {cells_alpha:>13}")
    print(f"  {'-'*34}  {'-'*10}  {'-'*13}")
    cost_diff = cost_alpha - cost_no_alpha
    cell_diff = cells_alpha - cells_no_alpha
    print(f"  {'Difference (with - without)':>34}  {cost_diff:>+10.2f}  {cell_diff:>+13}")