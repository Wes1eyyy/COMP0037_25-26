#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

import time

from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import full_scenario
from p1.high_level_actions import HighLevelActionType
from p1.high_level_environment import HighLevelEnvironment, PlannerType

if __name__ == '__main__':
    
    # Create the scenario
    airport_map, drawer_height = full_scenario()
    
    # Enable cell-type dependent traversability costs
    airport_map.set_use_cell_type_traversability_costs(True)
    
    # Get all rubbish bins
    all_rubbish_bins = airport_map.all_rubbish_bins()
    
    # Results storage
    results = []

    for planner_type, planner_name in [(PlannerType.BREADTH_FIRST, "BFS"),
                                        (PlannerType.DEPTH_FIRST, "DFS"),
                                        (PlannerType.DIJKSTRA, "Dijkstra"),
                                        (PlannerType.A_STAR, "A*")]:
        
        env = HighLevelEnvironment(airport_map, planner_type)
        env.show_graphics(False)
        env.show_verbose_graphics(False)

        # Teleport to start
        action = (HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0))
        env.step(action)

        total_cells_visited = 0
        total_path_cost = 0.0
        total_time = 0.0

        for rb in all_rubbish_bins:
            action = (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rb.coords())

            t0 = time.time()
            observation, reward, done, plan = env.step(action)
            elapsed = time.time() - t0

            cells_visited = plan.number_of_cells_visited
            path_cost = plan.path_travel_cost

            total_cells_visited += cells_visited
            total_path_cost += path_cost
            total_time += elapsed

            print(f"  {planner_name} -> bin {rb.coords()}: "
                  f"cells={cells_visited}, cost={path_cost:.2f}, time={elapsed:.4f}s")

        results.append({
            'name': planner_name,
            'total_cells': total_cells_visited,
            'total_cost': total_path_cost,
            'total_time': total_time,
            'num_bins': len(all_rubbish_bins),
        })

        print(f"\n{planner_name} TOTAL: cells={total_cells_visited}, "
              f"cost={total_path_cost:.2f}, time={total_time:.4f}s\n")

    # Summary table
    print("=" * 80)
    print(f"{'SUMMARY':^80}")
    print("=" * 80)
    print(f"{'Algorithm':>12} | {'Total Cells':>12} | {'Total Cost':>12} | {'Time (s)':>10} | {'Bins':>5}")
    print(f"{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*5}")
    for r in results:
        print(f"{r['name']:>12} | {r['total_cells']:>12d} | {r['total_cost']:>12.2f} | "
              f"{r['total_time']:>10.4f} | {r['num_bins']:>5d}")

    # Pairwise comparisons
    by_name = {r['name']: r for r in results}
    if 'Dijkstra' in by_name and 'A*' in by_name:
        d, a = by_name['Dijkstra'], by_name['A*']
        reduction = (d['total_cells'] - a['total_cells']) / d['total_cells'] * 100
        print(f"\nA* visited {reduction:.1f}% fewer cells than Dijkstra")
        print(f"Path costs: Dijkstra={d['total_cost']:.2f}, A*={a['total_cost']:.2f}")
        if abs(d['total_cost'] - a['total_cost']) < 1e-6:
            print("Path costs are IDENTICAL (admissible heuristic guarantees optimality)")
        else:
            print(f"Path cost difference: {abs(d['total_cost'] - a['total_cost']):.6f}")
    if 'BFS' in by_name:
        print(f"\nBFS total cost: {by_name['BFS']['total_cost']:.2f}, "
              f"cells: {by_name['BFS']['total_cells']}")
    if 'DFS' in by_name:
        print(f"DFS total cost: {by_name['DFS']['total_cost']:.2f}, "
              f"cells: {by_name['DFS']['total_cells']}")
    print("=" * 80)
    
