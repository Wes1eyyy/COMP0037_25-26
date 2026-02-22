'''
Created on 29 Jan 2022

@author: ucacsjj
'''

from .dynamic_programming_base import DynamicProgrammingBase

# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
   
    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):

        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3f:
    # Finish the implementation of the methods below.
    
    def _compute_optimal_value_function(self):

        # Get the environment and map
        environment = self._environment
        map = environment.map()

        iteration = 0

        while True:

            delta = 0

            # Sweep over all states
            for x in range(map.width()):
                for y in range(map.height()):

                    # Skip obstruction and terminal cells
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue

                    old_v = self._v.value(x, y)

                    # Find the maximum action-value over all 8 directional actions
                    best_v = float('-inf')

                    for a in range(8):
                        s_prime, r, p = environment.next_state_and_reward_distribution((x, y), a)

                        # Compute q(s, a) = sum over outcomes of p * (r + gamma * V(s'))
                        q = 0
                        for t in range(len(p)):
                            sc = s_prime[t].coords()
                            q += p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))

                        if q > best_v:
                            best_v = q

                    # Update the value function with the Bellman optimality backup
                    self._v.set_value(x, y, best_v)

                    # Track the maximum change across all states
                    delta = max(delta, abs(old_v - best_v))

            iteration += 1
            print(f'Value iteration step {iteration}, delta={delta}')

            # Update the drawer if available
            if self._value_drawer is not None:
                self._value_drawer.update()

            # Terminate if converged
            if delta < self._theta:
                break

            # Terminate if maximum iterations reached
            if iteration >= self._max_optimal_value_function_iterations:
                print('Maximum number of value iteration steps exceeded')
                break

    def _extract_policy(self):

        # Get the environment and map
        environment = self._environment
        map = environment.map()

        # For each non-obstruction, non-terminal state, pick the action
        # that maximises the action-value function under the converged V
        for x in range(map.width()):
            for y in range(map.height()):

                # Skip obstruction and terminal cells
                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                    continue

                best_action = 0
                best_q = float('-inf')

                for a in range(8):
                    s_prime, r, p = environment.next_state_and_reward_distribution((x, y), a)

                    # Compute q(s, a)
                    q = 0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        q += p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))

                    if q > best_q:
                        best_q = q
                        best_action = a

                self._pi.set_action(x, y, best_action)
