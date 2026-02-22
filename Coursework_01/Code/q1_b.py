#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import full_scenario
from p1.high_level_actions import HighLevelActionType
from p1.high_level_environment import HighLevelEnvironment, PlannerType


def run_planner_on_all_bins(airport_map, planner_type, show_graphics=True):
    """Run the given planner from (0,0) to every rubbish bin in sequence.
    Returns a list of (bin_number, goal_coords, path_cost, cells_visited)."""

    env = HighLevelEnvironment(airport_map, planner_type)
    env.show_graphics(show_graphics)
    env.show_verbose_graphics(False)

    env.step((HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0)))

    results = []
    for bin_number, rubbish_bin in enumerate(airport_map.all_rubbish_bins(), start=1):
        _, _, _, plan = env.step(
            (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
        )
        results.append((bin_number, rubbish_bin.coords(),
                        plan.path_travel_cost, plan.number_of_cells_visited))

        if show_graphics:
            screenshot_name = f'bin_{bin_number:02d}_{str(planner_type)}.pdf'
            env.search_grid_drawer().save_screenshot(screenshot_name)

    return results


def print_results(name, results):
    col = 56
    print(f"\n{'=' * col}")
    print(f"  Planner: {name}")
    print(f"{'=' * col}")
    print(f"  {'Bin':>3}  {'Goal':>10}  {'Path cost':>10}  {'Cells visited':>13}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*13}")

    total_cost = 0
    total_cells = 0
    for bin_num, coords, cost, cells in results:
        print(f"  {bin_num:>3}  {str(coords):>10}  {cost:>10.2f}  {cells:>13}")
        total_cost += cost
        total_cells += cells

    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*13}")
    print(f"  {'TOT':>3}  {'':>10}  {total_cost:>10.2f}  {total_cells:>13}")
    return total_cost, total_cells


if __name__ == '__main__':

    # Create the scenario
    airport_map, drawer_height = full_scenario()

    print(airport_map)

    # Draw the map (comment out if not needed)
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    airport_map_drawer.wait_for_key_press()

    # Q1b: Evaluate breadth-first and depth-first planners.
    summary = {}
    for planner_type, name in [
        (PlannerType.BREADTH_FIRST, 'Breadth-First'),
        (PlannerType.DEPTH_FIRST,   'Depth-First'),
    ]:
        results = run_planner_on_all_bins(airport_map, planner_type, show_graphics=True)
        total_cost, total_cells = print_results(name, results)
        summary[name] = (total_cost, total_cells)

    # Head-to-head summary
    col = 56
    print(f"\n{'=' * col}")
    print(f"  Summary comparison")
    print(f"{'=' * col}")
    print(f"  {'Planner':>14}  {'Total path cost':>16}  {'Total cells visited':>19}")
    print(f"  {'-'*14}  {'-'*16}  {'-'*19}")
    for name, (cost, cells) in summary.items():
        print(f"  {name:>14}  {cost:>16.2f}  {cells:>19}")