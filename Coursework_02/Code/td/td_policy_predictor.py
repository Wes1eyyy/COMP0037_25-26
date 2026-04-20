'''
Created on 8 Mar 2023

@author: ucacsjj
'''

from monte_carlo.episode_sampler import EpisodeSampler

from .td_algorithm_base import TDAlgorithmBase

class TDPolicyPredictor(TDAlgorithmBase):

    def __init__(self, environment):
        
        TDAlgorithmBase.__init__(self, environment)
        
        self._minibatch_buffer= [None]
                
    def set_target_policy(self, policy):        
        self._pi = policy        
        self.initialize()
        self._v.set_name("TDPolicyPredictor")
        
    def evaluate(self):
        
        episode_sampler = EpisodeSampler(self._environment)
        
        for episode in range(self._number_of_episodes):

            # Choose the start for the episode            
            start_x, start_a  = self._select_episode_start()
            self._environment.reset(start_x) 
            
            # Now sample it
            new_episode = episode_sampler.sample_episode(self._pi, start_x, start_a)

            # If we didn't terminate, skip this episode
            if new_episode.terminated_successfully() is False:
                continue
            
            # Update with the current episode
            self._update_value_function_from_episode(new_episode)
            
            # Pick several randomly from the experience replay buffer and update with those as well
            for _ in range(min(self._replays_per_update, self._stored_experiences)):
                episode = self._draw_random_episode_from_experience_replay_buffer()
                self._update_value_function_from_episode(episode)
                
            self._add_episode_to_experience_replay_buffer(new_episode)
            
    def _update_value_function_from_episode(self, episode):

        # Q1e:
        # TD(0) update: V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]
        # Walk through every step except the last terminal transition.
        for t in range(episode.number_of_steps() - 1):
            s_coords = episode.state(t).coords()
            s_next_coords = episode.state(t + 1).coords()
            r = episode.reward(t)

            v_s = self._v.value(s_coords[0], s_coords[1])
            v_s_next = self._v.value(s_next_coords[0], s_next_coords[1])

            new_v = v_s + self._alpha * (r + self._gamma * v_s_next - v_s)
            self._v.set_value(s_coords[0], s_coords[1], new_v)

