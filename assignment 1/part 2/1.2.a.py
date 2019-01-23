"""
GridWorld
"""
import itertools
from collections import defaultdict
import numpy as np


class GridWorld:
    def __init__(self, size, action_prob, gamma, delta):
        self.gamma = gamma
        self.action_prob = action_prob
        self.size = size
        self.delta = delta

        self.state_value = np.zeros((size, size))
        self.rewards = np.zeros((size, size))
        self.policy = defaultdict(list)

        self.mapping = np.arange(size ** 2).reshape(size, size)
        self.state_to_idx = defaultdict(int)
        self.idx_to_state = defaultdict(tuple)
        count = 0
        for i in range(size):
            for j in range(size):
                self.state_to_idx[(i, j)] = count
                self.idx_to_state[count] = (i, j)
                self.rewards[i][j] = 0
                count += 1

        # self.actions = ['L', 'R', 'U', 'D']
        self.num_actions = 4
        self.move = [np.array([0, -1]),
                     np.array([0, 1]),
                     np.array([-1, 0]),
                     np.array([1, 0])]
        # Set the rewards in the top left and right corners
        self.rewards[0][0] = 1
        self.rewards[0][size-1] = 10
        self.rewards = self.rewards.reshape((-1, ))
        # P(S' | S, a)
        self.prob_san = np.zeros(
            (self.size ** 2, self.num_actions, self.size ** 2))
        # P_pi(S' | S)
        self.prob_sn = np.zeros((self.size ** 2, self.size ** 2))
        # r_pi(S)
        self.rewards_pi = np.zeros((size**2, 1))

    # Make a valid transition to the next state
    def _generate_valid_move(self, current_state, action, step):
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
        next_state = np.asarray(list(current_state)) + step
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

        return tuple(next_state)

    # Calculates P(S' | S, a) based on the current policy
    def _generate_next_state_probs(self):
        print("Generating P(S' | S, a)")
        for s_idx in range(self.size**2):
            s_name = tuple(np.argwhere(self.mapping == s_idx)[0])
            for a_idx in range(self.num_actions):
                # For a given action get all the possible next states and calculate their
                # probabilities
                for move_idx, move_step in enumerate(self.move):
                    next_state = self._generate_valid_move(
                        tuple(s_name), move_idx, move_step)
                    ns_idx = self.state_to_idx[next_state]
                    # Probability p that the actual action taken is followed and a
                    # random action followed is probability (1-p)/3
                    if a_idx == move_idx:
                        self.prob_san[s_idx][a_idx][ns_idx] += self.action_prob
                    else:
                        self.prob_san[s_idx][a_idx][ns_idx] += (
                            1-self.action_prob)/3

    def _generate_transition_pi(self):
        print("Generating transition P_pi(S' | S)")
        # print(self.policy)
        for s_idx in range(self.size ** 2):
            temp_prob_sn = np.zeros(self.size ** 2)
            s_name = tuple(np.argwhere(self.mapping == s_idx)[0])
            # print(s_name)
            action_probs = self.policy[s_name]
            # print(action_probs)
            for a_idx, a_val in enumerate(action_probs):
                temp_prob_sn += np.array(a_val * self.prob_san[s_idx][a_idx])
            self.prob_sn[s_idx] = temp_prob_sn

    def _generate_reward_pi(self):
        print("Generating r_pi(S)")
        for s_idx in range(self.size ** 2):
            temp_reward_pi = 0
            s_name = tuple(np.argwhere(self.mapping == s_idx)[0])
            action_probs = self.policy[s_name]
            for a_idx, a_val in enumerate(action_probs):
                temp_reward_pi += a_val * \
                    np.dot(self.prob_san[s_idx][a_idx], self.rewards)
            self.rewards_pi[s_idx] = temp_reward_pi

    # Get random policy to start policy evaluation
    def _generate_random_policy(self):
        for row in range(self.size):
            for col in range(self.size):
                # Set a random action as policy
                self.policy[(row, col)] = np.eye(
                    1, self.num_actions, k=np.random.choice(self.num_actions)).squeeze()

    def _do_policy_evaluation(self):
        #counter = itertools.count()
        self.state_value = np.reshape(self.state_value, (-1, 1))
        while True:
            # print(next(counter))
            updated_state_value = self.rewards_pi + self.gamma * \
                np.matmul(self.prob_sn, self.state_value)
            if all(i < self.delta for i in np.abs(updated_state_value - self.state_value)):
                break
            self.state_value = updated_state_value

    def do_policy_iteration(self):
        # Start with a deterministic random policy
        self._generate_random_policy()
        # P(S' | S, a)
        self._generate_next_state_probs()
        while True:
            # P_pi(S' | S)
            self._generate_transition_pi()
            # r_pi(S)
            self._generate_reward_pi()
            # Calculate
            self._do_policy_evaluation()
            # Flag to check if the policy is stable or not
            is_stable = True
            for state, action_probs in self.policy.items():
                # Select the best action as per current policy
                old_action_idx = np.argmax(action_probs)
                s_idx = self.state_to_idx[state]
                q_values = []
                for a_idx, a_step in enumerate(self.move):
                    temp = np.dot(self.prob_san[s_idx][a_idx],
                                  self.rewards + self.gamma * self.state_value.squeeze())
                    q_values.append(temp)

                q_max_idx = np.argmax(q_values)
                # Update policy
                self.policy[state] = np.eye(
                    1, self.num_actions, k=q_max_idx).squeeze()

                # Check if the action is greedy or not
                if old_action_idx != q_max_idx:
                    is_stable = False

            if is_stable:
                break

    def get_state_value(self):
        return self.state_value.reshape((self.size, self.size))


# Parameters
WORLD_SIZES = [3]
ACTION_PROBS = [0.9]
GAMMAS = [0.9]

for SIZE in WORLD_SIZES:
    for ACTION_PROB in ACTION_PROBS:
        for GAMMA in GAMMAS:
            # Create grid
            grid = GridWorld(size=SIZE, action_prob=ACTION_PROB,
                             gamma=GAMMA, delta=1e-3)
            grid.do_policy_iteration()
            state_value = grid.get_state_value()
            print(state_value)
