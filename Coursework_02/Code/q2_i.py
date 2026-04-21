#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math
import random
import time
from pathlib import Path

import matplotlib
import numpy as np

from common.scenarios import slippy_corridor
from td.sarsa import SARSA
from td.q_learner import QLearner
from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from common.airport_map import MapCellType

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "report" / "Robotics_CW2-2" / "diagram"
NUM_OUTER_ITERATIONS = 40
EPISODES_PER_UPDATE = 32
ALPHA = 0.1
REPLAY_BUFFER_SIZE = 64
RANDOM_SEED = 7


ACTION_DELTAS = {
    LowLevelActionType.MOVE_RIGHT: (1, 0),
    LowLevelActionType.MOVE_UP_RIGHT: (1, 1),
    LowLevelActionType.MOVE_UP: (0, 1),
    LowLevelActionType.MOVE_UP_LEFT: (-1, 1),
    LowLevelActionType.MOVE_LEFT: (-1, 0),
    LowLevelActionType.MOVE_DOWN_LEFT: (-1, -1),
    LowLevelActionType.MOVE_DOWN: (0, -1),
    LowLevelActionType.MOVE_DOWN_RIGHT: (1, -1),
}


class SlippyLowLevelEnvironment(LowLevelEnvironment):

    def next_state_and_reward_distribution(self, s, a, print_cell=False):

        current_cell = self._airport_map.cell(s[0], s[1])
        p_slip = current_cell.p_slip()

        s_prime = []
        r = []
        p = []

        if a == LowLevelActionType.TERMINATE:
            if current_cell.is_terminal() is True:
                return [None], [current_cell.params()], [1]

        if a == LowLevelActionType.NONE:
            return [current_cell], [-1], [1]

        if p_slip > 0:
            s_prime.append(current_cell)
            r.append(-1)
            p.append(p_slip)

        for i in range(-1, 2):
            if i == 0:
                pr = (1 - p_slip) * self._p
            else:
                pr = (1 - p_slip) * self._q

            idx = a + i
            if idx > 7:
                idx = -3

            delta = self._driving_deltas[idx]
            new_x = s[0] + delta[0]
            new_y = s[1] + delta[1]

            if (new_x < 0) or (new_x >= self._airport_map.width()) \
                or (new_y < 0) or (new_y >= self._airport_map.height()):
                s_prime.append(current_cell)
                r.append(-1)
            else:
                new_cell = self._airport_map.cell(new_x, new_y)
                if new_cell.is_obstruction():
                    s_prime.append(current_cell)
                    if new_cell.cell_type() is MapCellType.BAGGAGE_CLAIM:
                        r.append(-10)
                    else:
                        r.append(-1)
                else:
                    s_prime.append(new_cell)
                    r.append(-self._airport_map.compute_transition_cost(current_cell.coords(), new_cell.coords()))

            p.append(pr)

        return s_prime, r, p


def _extract_value_grid(value_function, airport_map):
    width = airport_map.width()
    height = airport_map.height()
    grid = np.full((height, width), np.nan)

    for x in range(width):
        for y in range(height):
            cell = airport_map.cell(x, y)
            if cell.is_obstruction():
                continue
            grid[height - 1 - y, x] = value_function.value(x, y)

    return grid


def _plot_value_function(value_function, airport_map, output_path, title):
    grid = _extract_value_grid(value_function, airport_map)
    fig, ax = plt.subplots(figsize=(10, 3))
    image = ax.imshow(grid, cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="V(s)")

    height, width = grid.shape
    for x in range(width):
        for y in range(height):
            cell = airport_map.cell(x, height - 1 - y)
            if cell.p_slip() > 0:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.0))
            if np.isfinite(grid[y, x]):
                ax.text(x, y, f"{grid[y, x]:.1f}", ha="center", va="center", color="white", fontsize=6)

    ax.set_title(title)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_policy(policy, airport_map, output_path, title):
    width = airport_map.width()
    height = airport_map.height()
    fig, ax = plt.subplots(figsize=(10, 3))

    for x in range(width):
        for y in range(height):
            cell = airport_map.cell(x, y)
            display_y = height - 1 - y

            if cell.is_obstruction():
                ax.add_patch(plt.Rectangle((x - 0.5, display_y - 0.5), 1, 1, color="black"))
                continue

            if cell.p_slip() > 0:
                ax.add_patch(plt.Rectangle((x - 0.5, display_y - 0.5), 1, 1, color="#d9d9d9", alpha=0.6))

            action = policy.greedy_optimal_action(x, y)
            if action == LowLevelActionType.TERMINATE:
                ax.plot(x, display_y, "ro", markersize=5)
            elif action == LowLevelActionType.NONE:
                ax.plot(x, display_y, "ko", markersize=3)
            else:
                dx, dy = ACTION_DELTAS[action]
                ax.arrow(
                    x,
                    display_y,
                    0.35 * dx,
                    -0.35 * dy,
                    head_width=0.16,
                    head_length=0.12,
                    fc="tab:red",
                    ec="tab:red",
                    length_includes_head=True,
                )

    ax.set_title(title)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, linewidth=0.5, color="lightgray")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_controller(learner_cls, value_pdf_name, policy_pdf_name):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    airport_map, _ = slippy_corridor()
    env = SlippyLowLevelEnvironment(airport_map)
    pi = env.initial_policy()
    pi.set_epsilon(1)

    learner = learner_cls(env)
    learner.set_alpha(ALPHA)
    learner.set_experience_replay_buffer_size(REPLAY_BUFFER_SIZE)
    learner.set_number_of_episodes(EPISODES_PER_UPDATE)
    learner.set_initial_policy(pi)

    iteration_times = []
    for i in range(NUM_OUTER_ITERATIONS):
        start_time = time.perf_counter()
        learner.find_policy()
        iteration_times.append(time.perf_counter() - start_time)
        pi.set_epsilon(1 / math.sqrt(1 + 0.25 * i))

    _plot_value_function(
        learner.value_function(),
        airport_map,
        OUTPUT_DIR / value_pdf_name,
        f"{learner.value_function().name()} on Slippy Corridor",
    )
    _plot_policy(
        learner.policy(),
        airport_map,
        OUTPUT_DIR / policy_pdf_name,
        f"{learner.policy().name()} on Slippy Corridor",
    )

    sample_states = {
        "left_entry": (4, 4),
        "slip_centre": (10, 4),
        "goal_approach": (18, 4),
    }
    sampled_values = {
        key: learner.value_function().value(coords[0], coords[1])
        for key, coords in sample_states.items()
    }

    return {
        "mean_iteration_time_ms": 1000 * sum(iteration_times) / len(iteration_times),
        "max_iteration_time_ms": 1000 * max(iteration_times),
        "sampled_values": sampled_values,
    }


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sarsa_results = train_controller(
        SARSA,
        "q2_i_sarsa_value.pdf",
        "q2_i_sarsa_policy.pdf",
    )
    qlearning_results = train_controller(
        QLearner,
        "q2_i_qlearning_value.pdf",
        "q2_i_qlearning_policy.pdf",
    )

    print("Q2i results")
    print(f"SARSA mean iteration time (ms): {sarsa_results['mean_iteration_time_ms']:.3f}")
    print(f"SARSA max iteration time (ms): {sarsa_results['max_iteration_time_ms']:.3f}")
    print(f"SARSA sampled values: {sarsa_results['sampled_values']}")
    print(f"Q-learning mean iteration time (ms): {qlearning_results['mean_iteration_time_ms']:.3f}")
    print(f"Q-learning max iteration time (ms): {qlearning_results['max_iteration_time_ms']:.3f}")
    print(f"Q-learning sampled values: {qlearning_results['sampled_values']}")
        
