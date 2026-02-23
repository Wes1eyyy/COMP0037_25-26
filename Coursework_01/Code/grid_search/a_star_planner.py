'''
Created on 2 Jan 2022

@author: ucacsjj
'''

import math

from .dijkstra_planner import DijkstraPlanner
from .occupancy_grid import OccupancyGrid
from .search_grid import SearchGridCell


class AStarPlanner(DijkstraPlanner):
    def __init__(self, occupancy_grid: OccupancyGrid):
        DijkstraPlanner.__init__(self, occupancy_grid)

    # Q2d:
    # Complete implementation of A*.
    # Override push to use f(n) = g(n) + h(n) as the priority.

    def push_cell_onto_queue(self, cell: SearchGridCell):
        # Compute g(n): cumulative path cost from start
        if cell.parent is not None:
            cell.path_cost = cell.parent.path_cost + \
                self.compute_l_stage_additive_cost(cell.parent, cell)

        # Compute h(n): Euclidean distance to goal (admissible heuristic)
        cell_coords = cell.coords()
        goal_coords = self.goal.coords()
        dx = cell_coords[0] - goal_coords[0]
        dy = cell_coords[1] - goal_coords[1]
        h = math.sqrt(dx * dx + dy * dy)

        # Priority = f(n) = g(n) + h(n)
        f = cell.path_cost + h
        self.priority_queue.put((f, cell))

    def resolve_duplicate(self, cell: SearchGridCell, parent_cell: SearchGridCell):
        # Edge relaxation using g-cost, but re-queue with f = g + h
        new_cost = parent_cell.path_cost + \
            self.compute_l_stage_additive_cost(parent_cell, cell)
        if new_cost < cell.path_cost:
            cell.path_cost = new_cost
            cell.set_parent(parent_cell)

            # Recompute h and queue with f priority
            cell_coords = cell.coords()
            goal_coords = self.goal.coords()
            dx = cell_coords[0] - goal_coords[0]
            dy = cell_coords[1] - goal_coords[1]
            h = math.sqrt(dx * dx + dy * dy)
            self.priority_queue.put((new_cost + h, cell))
