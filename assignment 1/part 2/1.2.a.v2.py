"""
GridWorld
"""
import copy
import itertools
from collections import defaultdict
import numpy as np


class GridWorld:
    """
    GridWorld
    """

    def __init__(self, size, action_prob, gamma):
        self.size = size
        self.gamma = gamma
        self.action_prob = action_prob
        self.mapping = np.arange(size ** 2).reshape(size, size)

        # Initialize state rewards
        self.rewards = np.zeros((size, size))
        # Set the rewards in the top left and right corners
        self.rewards[0][0] = 1
        self.rewards[0][size-1] = 10
        self.rewards = self.rewards.reshape((-1, ))
        # Define the policy and utility arrays to get convert
        # state -> idx and reverse

        # Actions and corresponding transitions = ['L', 'R', 'U', 'D']
        self.num_actions = 4
        self.move = [np.array([0, -1]),
                     np.array([0, 1]),
                     np.array([-1, 0]),
                     np.array([1, 0])]
        # Initialize P(S' | S, a). We only need to call this once as it is
        # the dynamics of the environment and need not be evaluated multiple
        # times during the policy, value and modified policy iteration.
        self.prob_san = np.zeros(
            (self.size ** 2, self.num_actions, self.size ** 2))
        self._generate_next_state_probs()

    # Calculates P(S' | S, a) based on the current policy
    def _generate_next_state_probs(self):
        print("Generating P(S' | S, a)")
        for s_idx in range(self.size**2):
            for a_idx in range(self.num_actions):
                # For a given action get all the possible next states and calculate their
                # probabilities
                for move_idx, move_step in enumerate(self.move):
                    ns_idx = self._generate_valid_move(
                        s_idx, move_idx, move_step)

                    # Probability p that the actual action taken is followed and a
                    # random action followed is probability (1-p)/3
                    if a_idx == move_idx:
                        self.prob_san[s_idx][a_idx][ns_idx] += self.action_prob
                    else:
                        self.prob_san[s_idx][a_idx][ns_idx] += (
                            1-self.action_prob)/(len(self.move)-1)

    # Make a valid transition to the next state
    def _generate_valid_move(self, current_state_idx, action, step):
        """
        Input:
        State: tuple
        (i, j)

        Action: int
        0: Left
        1: Right
        2: Up
        3: Down
        """
        current_state = np.argwhere(self.mapping == current_state_idx)[0]
        next_state = np.asarray(current_state) + step
        # Left
        if action == 0 and next_state[1] < 0:
            next_state = current_state
        # Right
        elif action == 1 and next_state[1] > self.size-1:
            next_state = current_state
        # Up
        elif action == 2 and next_state[0] < 0:
            next_state = current_state
        # Down
        elif action == 3 and next_state[0] > self.size-1:
            next_state = current_state

        return self.mapping[next_state[0]][next_state[1]]


class Agent:
    def __init__(self, grid, delta):
        self.delta = delta
        self.policy = defaultdict(list)
        self.mapping = np.arange(grid.size ** 2).reshape(grid.size, grid.size)
        self.state_to_idx = defaultdict(int)
        self.idx_to_state = defaultdict(tuple)
        # Initialize the V_0(S) = 0, for all S
        self.state_value = np.zeros(grid.size ** 2)

        count = 0
        for i in range(grid.size):
            for j in range(grid.size):
                self.state_to_idx[(i, j)] = count
                self.idx_to_state[count] = (i, j)
                count += 1

        # P_pi(S' | S)
        self.prob_sn_pi = np.zeros((grid.size**2, grid.size**2))
        # r_pi(S)
        self.rewards_pi = np.zeros(grid.size**2)

    # Get random policy to start policy evaluation
    def _generate_random_policy(self, grid):
        for state in range(grid.size ** 2):
            self.policy[state] = np.eye(grid.num_actions)[
                np.random.choice(grid.num_actions)]

    # P_pi(S' | S)
    def _generate_transition_pi(self, grid):
        print("Generating transition P_pi(S' | S)")
        for s_idx in range(grid.size ** 2):
            temp_prob_sn = np.zeros(grid.size ** 2)
            action_probs = self.policy[s_idx]
            for a_idx, a_val in enumerate(action_probs):
                temp_prob_sn += np.array(a_val * grid.prob_san[s_idx][a_idx])
            self.prob_sn_pi[s_idx] = temp_prob_sn

    # r_pi(S' | S)
    def _generate_reward_pi(self, grid):
        print("Generating r_pi(S)")
        for s_idx in range(grid.size ** 2):
            temp_reward_pi = 0
            action_probs = self.policy[s_idx]
            for a_idx, a_val in enumerate(action_probs):
                temp_reward_pi += a_val * \
                    np.dot(grid.prob_san[s_idx][a_idx], grid.rewards)
            self.rewards_pi[s_idx] = temp_reward_pi

    def _do_policy_evaluation(self, grid, value_iter):
        counter = itertools.count()
        self.state_value = np.zeros(self.state_value.shape)
        while True:
            updated_state_value = self.rewards_pi + grid.gamma * \
                np.matmul(self.prob_sn_pi, self.state_value)

            if all(i < self.delta for i in np.abs(updated_state_value - self.state_value)):
                break

            self.state_value = updated_state_value

            if next(counter) == value_iter:
                print("Early exit")
                break

    def do_policy_iteration(self, grid, value_iter=-1):
        # Start with a deterministic random policy
        self._generate_random_policy(grid)
        print(self.policy)
        while True:
            # P_pi(S' | S)
            self._generate_transition_pi(grid)
            # r_pi(S)
            self._generate_reward_pi(grid)
            print(self.rewards_pi)
            # Calculate value function
            self._do_policy_evaluation(grid, value_iter)
            # Flag to check if the policy is stable or not
            is_stable = True
            for s_idx, action_probs in self.policy.items():
                # Select the best action as per current policy
                old_action_idx = np.argmax(action_probs)
                q_values = []
                for a_idx, _ in enumerate(grid.move):
                    temp = np.dot(grid.prob_san[s_idx][a_idx],
                                  grid.rewards + grid.gamma * self.state_value)
                    q_values.append(temp)

                q_max_idx = np.argmax(q_values)
                # Update policy
                self.policy[s_idx] = np.eye(grid.num_actions)[q_max_idx]

                # Check if the action is greedy or not
                if old_action_idx != q_max_idx:
                    is_stable = False

            if is_stable:
                break

        return self.state_value.reshape((grid.size, grid.size)), self.policy

    def do_value_iteration(self, grid):
        while True:
            old_state_value = copy.deepcopy(self.state_value)
            # Sweep through state-space and update state-value using Bellman optimality
            # equation
            for s_idx, s_val in enumerate(self.state_value):
                q_values = []
                for a_idx, a_step in enumerate(grid.move):
                    q_values.append(np.dot(grid.prob_san[s_idx][a_idx],
                                           grid.rewards + grid.gamma * self.state_value))
                q_max_idx = np.argmax(q_values)
                self.state_value[s_idx] = q_values[q_max_idx]
                self.policy[tuple(np.argwhere(self.mapping == s_idx)[0])] = np.eye(
                    grid.num_actions)[q_max_idx]

            if all(i < self.delta for i in np.abs(old_state_value - self.state_value)):
                break

        return self.state_value.reshape((grid.size, grid.size)), self.policy


# Parameters
WORLD_SIZES = [3]
ACTION_PROBS = [0.9]
GAMMAS = [0.9]

for SIZE in WORLD_SIZES:
    for ACTION_PROB in ACTION_PROBS:
        for GAMMA in GAMMAS:
            # Create grid
            Grid = GridWorld(size=SIZE, action_prob=ACTION_PROB,
                             gamma=GAMMA)
            agent = Agent(grid=Grid, delta=0.0001)
            state_value, policy = agent.do_policy_iteration(Grid)
            # state_value, policy = agent.do_value_iteration(Grid)
            # state_value, policy = agent.do_policy_iteration(Grid, value_iter=5)

            print(state_value)
            print(policy)
