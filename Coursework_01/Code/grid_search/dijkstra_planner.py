'''
Created on 2 Jan 2022

@author: ucacsjj
'''

from queue import PriorityQueue

from .occupancy_grid import OccupancyGrid
from .planner_base import PlannerBase
from .search_grid import SearchGridCell, SearchGridCellLabel


class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.

    def __init__(self, occupancy_grid: OccupancyGrid):
        PlannerBase.__init__(self, occupancy_grid)
        self.priority_queue = PriorityQueue()  # type: ignore

    # Q1d:
    # Modify this class to finish implementing Dijkstra

    def push_cell_onto_queue(self, cell: SearchGridCell):
        # Compute the cumulative path cost to this cell via its parent and
        # store it on the cell so resolve_duplicate can compare costs later.
        if cell.parent is not None:
            cell.path_cost = cell.parent.path_cost + \
                self.compute_l_stage_additive_cost(cell.parent, cell)
        self.priority_queue.put((cell.path_cost, cell))

    def is_queue_empty(self) -> bool:
        return self.priority_queue.empty()

    def pop_cell_from_queue(self) -> SearchGridCell:
        # Lazy deletion: entries for cells already finalised (DEAD) are stale
        # duplicates inserted by resolve_duplicate. Skip them.
        while not self.priority_queue.empty():
            _, cell = self.priority_queue.get()
            if cell.label() != SearchGridCellLabel.DEAD:
                return cell

    def resolve_duplicate(self, cell: SearchGridCell, parent_cell: SearchGridCell):
        # Edge relaxation: if a cheaper path to cell is found through
        # parent_cell, update the cost and parent and re-queue the cell.
        new_cost = parent_cell.path_cost + \
            self.compute_l_stage_additive_cost(parent_cell, cell)
        if new_cost < cell.path_cost:
            cell.path_cost = new_cost
            cell.set_parent(parent_cell)
            self.priority_queue.put((new_cost, cell))