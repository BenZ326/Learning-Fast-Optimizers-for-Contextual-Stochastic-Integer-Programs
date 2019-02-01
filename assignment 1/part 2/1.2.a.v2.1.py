""" Gridworld MDP solver

We are given a stochastic Gridworld of size nxn with terminal states in the
upper right and upper left corner with a reward of +10 and +1 in respectively.
Rewards in all other states are zero. Whenever we take an action A from state
S in the Gridworld, we move to the desired next state S' with a probability p
and a random state with a probability 1-p. The goal is to find an optimal way
to behave in this Gridworld such that we have accumulated maximum rewards when
we reach any of the terminal states. Behave, here means the action which an
external agent should take, or the way in which it should behave, when it is
in some particular state of Gridworld. To formalize the notion of behaviour in
the context of Gridworld, we introduce one more concept called the state value
function. We can query this function to get a numerical estimate of how good it
is to be in queried state. Thus, when we have value estimate for all the states
of Gridworld, we define the optimal behvior in a state as the action which takes
the agent to state with better value. For finding the value of state, we use
Dynamic Programming as an iterative solution finding technique to find approximate
solution of the state values.

This script demonstrates the use of Dynamic Programming to find an approximation
of the true value function. We use this value functions to extract a policy
for behaving optimally in the Gridworld.
"""
# Author: Rahul Patel and Xiangyi Zhang
# Version: 1.0.0
# Date: 29/01/2019

import copy
import itertools
from collections import defaultdict
import numpy as np


class Gridworld:
    """
    A class used to represent the Gridworld

    ...
    Attributes
    ----------
    size : int
        The size of Gridworld (size x size)
    action_prob : int
        Probability of moving to desired next state by taking some action in the
        current state
    mapping : 2d array
        Maps the 2d representation of state to a one dimensional representation
        of states. The state [i][j] will be mapped to mapping[i][j]
    rewards : 1d array
        Store the rewards obtained by entering in a state
    num_actions : int
        Total number of actions allowed from any state
    move : list
        Stores moves indexed by actions
    prob_san : 3d array
        Store the transition probabilities P(S' | S, a)

    Methods
    -------
    _generate_valid_next_state(current_state_idx, action, step)
        Generates valid next states, when an agent performs some action in a
        state by preventing it from getting outside the grid

    _generate_next_state_probs()
        Finds the distribution over possible next states given the current state
        and action taken i.e. P(S' | S, a)
    """

    def __init__(self, size, action_prob):
        """
        Parameters
        ----------
        size : int
            The size of the Gridworld
        action_prob : int
            Probability of moving to desired next state by taking some action in the
            current state
        """
        self.size = size
        self.action_prob = action_prob
        self.mapping = np.arange(size ** 2).reshape(size, size)
        self.rewards = np.zeros(size ** 2)
        np.put(self.rewards, [0, size-1], [1.0, 10.0])

        self.num_actions = 4
        # O : Left
        # 1 : Right
        # 2 : Up
        # 3 : Down
        self.move = [np.array([0, -1]),
                     np.array([0, 1]),
                     np.array([-1, 0]),
                     np.array([1, 0])]

        # Initialize P(S' | S, a) with zeros
        self.prob_san = np.zeros(
            (self.size ** 2, self.num_actions, self.size ** 2))
        self._generate_next_state_probs()

    # Make a valid transition to the next state
    def _generate_valid_next_state(self, current_state_idx, action, step):
        """
        Generates valid next states, when an agent performs some action in a
        state by preventing it from getting outside the grid

        Parameters
        ----------
            current_state_idx : int
                The id of the current state
            action : int
                The id of the action taken
            step : 1d array
                The transition to perform from the current state

        Returns
        -------
            The next valid state. Keeps the agent in the current state if the
            agent's action might him out of the Gridworld
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

    # Calculate P(S' | S, a)
    def _generate_next_state_probs(self):
        """Finds the distribution over possible next states given the current state
        and action taken i.e. P(S' | S, a)
        """
        # print("Generating P(S' | S, a)")
        for s_idx in range(self.size**2):
            for a_idx in range(self.num_actions):
                # For a given action get all the possible next states and calculate their
                # probabilities
                for move_idx, move_step in enumerate(self.move):
                    ns_idx = self._generate_valid_next_state(
                        s_idx, move_idx, move_step)
                    # Environment moves the agent in the desired direction with probability p
                    # and a random next state state with probability 1-p
                    # TODO: Confirm with TA

                    # if a_idx == move_idx:
                    #     self.prob_san[s_idx][a_idx][ns_idx] += self.action_prob
                    # else:
                    #     self.prob_san[s_idx][a_idx][ns_idx] += (
                    #         1-self.action_prob)/(len(self.move)-1)

                    self.prob_san[s_idx][a_idx][ns_idx] += (1 -
                                                            self.action_prob)/len(self.move)
                    if a_idx == move_idx:
                        self.prob_san[s_idx][a_idx][ns_idx] += self.action_prob


class Agent:
    """
    Agent class which interact with the Gridworld and learns an optimal way
    to behave in it

    Attributes
    ----------
    gamma : int
        Discount factor used to weight the future rewards
    delta : int
        Tolerance used at check termination
    policy : dict(list)
        Store the best action for all states
    state_value : 1d array
        Store state value estimates of all states
    state_mask : 1d array
        Binary array, to set values of terminal state to zero
    prob_sn_pi : 2d array
        Store distribution of over possible next states given the current state,
        following policy pi -- P_pi(S' | S)
    prob_reward_pi : 1d array
        Store expected reward obtained in state S following policy pi -- r_pi(S)

    Methods
    -------
    _generate_random_policy(grid)
        Generate a random policy

    _generate_transition_pi(grid)
        Generate a distribution over possible next state from the current state
        under policy pi

    _generate_reward_pi(grid)
        Generate expected rewards obtained using the current policy pi

    _do_policy_evaluation(grid, eval_steps)
        Find the state values under the current policy

    do_policy_iteration(grid)
        Iteratively find the optimal policy by performing policy evaluation
        and policy improvement

    do_value_iteration(grid)
        Iteratively find the optimal policy by performing value iteration

    do_modified_policy_iteration(grid, eval_steps)
        Iteratively find the optimal policy by performing policy evaluation
        for eval_steps
    """

    def __init__(self, grid, delta, gamma):
        """
        Parameters
        ----------
        grid : object <Gridworld>
        delta : int
            Tolerance used at check termination
        gamma : int
            Discount factor used to weight the future rewards
        """
        self.gamma = gamma
        self.delta = delta
        self.policy = defaultdict(list)
        # Initialize the V_0(S) = 0, for all S
        self.state_value = np.zeros(grid.size ** 2)
        self.state_mask = np.ones(grid.size ** 2)
        np.put(self.state_mask, [0, grid.size-1], [0, 0])
        # P_pi(S' | S)
        self.prob_sn_pi = np.zeros((grid.size**2, grid.size**2))
        # r_pi(S)
        self.reward_pi = np.zeros(grid.size**2)

    # Generates random policy
    def _generate_random_policy(self, grid):
        """Generate a random policy and set the attribute self.policy

        Parameters
        ----------
            grid : object <Gridworld>
        """
        for state in range(grid.size ** 2):
            self.policy[state] = np.eye(grid.num_actions)[
                np.random.choice(grid.num_actions)]

    # Fixed random policy
    # def _generate_random_policy(self, grid):
    #     for state in range(grid.size ** 2):
    #         self.policy[state] = np.eye(grid.num_actions)[1]

    # Generate P_pi(S' | S)
    def _generate_transition_pi(self, grid):
        """Generate a distribution over possible next state from the current state
        under policy pi and set the attribute self.prob_sn

        Parameters
        ----------
            grid : object <Gridworld>
        """
        # print("Generating transition P_pi(S' | S)")
        for s_idx in range(grid.size ** 2):
            temp_prob_sn = np.zeros(grid.size ** 2)
            action_probs = self.policy[s_idx]
            for a_idx, a_val in enumerate(action_probs):
                temp_prob_sn += np.array(a_val * grid.prob_san[s_idx][a_idx])
            self.prob_sn_pi[s_idx] = temp_prob_sn

    # Generate r_pi(S)
    def _generate_reward_pi(self, grid):
        """Generate expected rewards obtained using the current policy pi and set
        the attribute self.reward_pi

        Parameters
        ----------
            grid : object <Gridworld>
        """
        # print("Generating r_pi(S)")
        for s_idx in range(grid.size ** 2):
            temp_reward_pi = 0
            action_probs = self.policy[s_idx]
            for a_idx, a_val in enumerate(action_probs):
                temp_reward_pi += a_val * \
                    np.dot(grid.prob_san[s_idx][a_idx], grid.rewards)
            self.reward_pi[s_idx] = temp_reward_pi

    def _do_policy_evaluation(self, grid, eval_steps=-1):
        """Find the state values under the current policy and set the attribute
        self.state_value

        Parameters
        ----------
            grid : object <Gridworld>

            eval_steps : int
                Number of itertions to perform policy evaluation
        """
        counter = itertools.count()
        should_stop = False
        while True:
            # Update state values
            updated_state_value = self.reward_pi + self.gamma * \
                np.matmul(self.prob_sn_pi, self.state_value)
            # Apply mask on terminal states
            updated_state_value = np.multiply(
                updated_state_value, self.state_mask)
            # Check stopping criterion
            max_diff = max(np.abs(updated_state_value - self.state_value))
            if max_diff < self.delta:
                should_stop = True
            if next(counter) == eval_steps-1:
                should_stop = True
            # Update the state values
            self.state_value = copy.deepcopy(updated_state_value)
            # Terminate if necessary
            if should_stop:
                break

    def do_policy_iteration(self, grid):
        """Iteratively find the optimal policy by performing policy evaluation
        and policy improvement

        Parameters
        ----------
            grid : object <Gridworld>

        Returns
        -------
            state_value, policy
        """
        counter = itertools.count()
        # Start with a deterministic random policy
        self._generate_random_policy(grid)
        self.state_value = np.zeros(self.state_value.shape)
        while True:
            # P_pi(S' | S)
            self._generate_transition_pi(grid)
            # r_pi(S)
            self._generate_reward_pi(grid)
            # Calculate value function
            self._do_policy_evaluation(grid)
            # Policy Improvement
            is_stable = True
            # changed_states = []
            self.state_value = np.around(self.state_value, 8)
            for s_idx, action_probs in self.policy.items():
                # Select the best action as per current policy
                old_action_idx = np.argmax(action_probs)
                # Find the maximizing action
                q_values = []
                for a_idx, _ in enumerate(grid.move):
                    q_values.append(np.dot(grid.prob_san[s_idx][a_idx],
                                           grid.rewards + self.gamma * self.state_value))
                # q_values = np.around(q_values, 8)
                q_max_idx = np.argmax(q_values)
                # Update policy
                self.policy[s_idx] = np.eye(grid.num_actions)[q_max_idx]
                # Check if the action is greedy or not
                if old_action_idx != q_max_idx:
                    # changed_states.append(np.argwhere(grid.mapping == s_idx))
                    is_stable = False
            # print(changed_states)
            visualize_policy(self.policy, grid.size)
            if is_stable:
                break

        return self.state_value.reshape((grid.size, grid.size)), self.policy

    def do_modified_policy_iteration(self, grid, eval_steps=5):
        """Iteratively find the optimal policy by performing policy evaluation
        and policy improvement

        Parameters
        ----------
            grid : object <Gridworld>

        Returns
        -------
            state_value, policy
        """
        self._generate_random_policy(grid)
        target_state_value = np.zeros(self.state_value.shape)
        while True:
            # P_pi(S' | S)
            self._generate_transition_pi(grid)
            # r_pi(S)
            self._generate_reward_pi(grid)
            # Evaluate the policy for k steps
            self._do_policy_evaluation(grid, eval_steps)
            for s_idx, action_probs in self.policy.items():
                if s_idx in [0, grid.size - 1]:
                    continue
                q_values = []
                for a_idx, _ in enumerate(grid.move):
                    q_values.append(np.dot(grid.prob_san[s_idx][a_idx],
                                           grid.rewards + self.gamma * self.state_value))

                q_max_idx = np.argmax(q_values)
                # Update policy. However, we do not check the stability of the policy
                # as an evaluation criterion.
                self.policy[s_idx] = np.eye(grid.num_actions)[q_max_idx]
                target_state_value[s_idx] = q_values[q_max_idx]

            # max (|| V_n - V_n-1 ||) < delta
            # print("Target state value")
            # print(target_state_value.reshape((grid.size, grid.size)))
            # print("State value")
            # print(self.state_value.reshape((grid.size, grid.size)))
            visualize_policy(self.policy, grid.size)
            if max(np.abs(self.state_value - target_state_value)) < self.delta:
                break

        return self.state_value.reshape((grid.size, grid.size)), self.policy

    def do_value_iteration(self, grid):
        """Iteratively find the optimal policy by performing value iteration

        Parameters
        ----------
            grid : object <Gridworld>

        Returns
        -------
            state_value, policy
        """
        terminal_states = [0, grid.size-1]
        self.state_value = np.zeros(self.state_value.shape)
        while True:
            visualize_policy(self.policy, grid.size)
            old_state_value = self.state_value.copy()
            # Sweep through state-space and update state-value using Bellman optimality
            # equation
            for s_idx, s_val in enumerate(self.state_value):
                q_values = []
                for a_idx, a_step in enumerate(grid.move):
                    q_values.append(np.dot(grid.prob_san[s_idx][a_idx],
                                           grid.rewards + self.gamma * old_state_value))
                q_max_idx = np.argmax(q_values)
                self.policy[s_idx] = np.eye(grid.num_actions)[q_max_idx]
                self.state_value[s_idx] = q_values[q_max_idx] \
                    if s_idx not in terminal_states else 0.0

            # Check stopping criterion
            if max(np.abs(old_state_value - self.state_value)) < self.delta:
                break

        return self.state_value.reshape((grid.size, grid.size)), self.policy


def visualize_policy(policy, size):
    """
    Visualize policy

    Parameters
    ----------
        policy : dict(list)
            Policy to be visualized
        size : int
            Size of the GridWorld
    """
    symbol = {0: '\u2190', 1: '\u2192', 2: '\u2191', 3: '\u2193', 4: '\u25A0'}

    size = int(len(policy)/size)

    for i in range(size):
        col = ""
        for j in range(size):
            action = np.argwhere(policy[i*size + j] == 1)[0][0]
            col += " " + symbol[action] + " "
        print(col)


# Parameters
WORLD_SIZES = [50]
ACTION_PROBS = [0.9]

for SIZE in WORLD_SIZES:
    for ACTION_PROB in ACTION_PROBS:
        print("SIZE {}".format(SIZE))
        print("Action Prob {}".format(ACTION_PROB))
        # Create grid
        grid = Gridworld(size=SIZE, action_prob=ACTION_PROB)
        agent = Agent(grid=grid, delta=1e-4, gamma=0.9)
        state_value, policy = agent.do_policy_iteration(grid=grid)
        # state_value, policy = agent.do_value_iteration(grid=grid)
        # state_value, policy = agent.do_modified_policy_iteration(
        #     grid, eval_steps=1)
        print(state_value)
        visualize_policy(policy, grid.size)
