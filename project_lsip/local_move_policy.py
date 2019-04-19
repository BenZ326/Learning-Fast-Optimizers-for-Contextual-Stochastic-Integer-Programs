from ActorCriticNetwork import ActorCriticNetwork
import torch as T
from torch.distributions.categorical import Categorical
import numpy as np


class A2CLocalMovePolicy:
    """
    A local move policy based on A2C
    """

    def __init__(self, args, initial_state=None, num_epochs=1000, num_local_move=2, gamma=0.99, beta_entropy=0.0001, lr_a2c=1e-4):
        self.initial_state = initial_state
        # Hyperparams
        self.num_epochs = num_epochs
        self.num_local_move = num_local_move
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.lr_a2c = lr_a2c

        self.a2c = ActorCriticNetwork(args)
        self.optimizer = T.optim.Adam(self.a2c.parameters(), lr=self.lr_a2c)

    def _reset_buffers(self):
        self.reward_buffer = list()
        self.state_buffer = list()
        self.action_buffer = list()
        self.state_value_buffer = list()
        self.action_prob_buffer = list()
        self.action_probs_buffer = list()

    def _select_action_and_record_state_value(self, state):
        """
        Select action based on the current state. Convert the input state to
        a torch tensor and pass it through the ActorCriticNetwork to get the
        probability distribution over the actions. Sample some action from
        this distribution and return

        Arguments:
        ----------
            state (numpy array) : A vector representing the current state
        """
        state_representation = T.from_numpy(
            state.get_representation()).float()

        action_probs, state_value = self.a2c.get_action_probs_and_state_value(
            state_representation)
        self.state_value_buffer.append(state_value.item())
        self.action_probs_buffer.append(action_probs)

        action = Categorical(probs=action_probs).sample()
        self.action_prob_buffer.append(action_probs[action.item()])

        return action.item()

    def _optimize_model(self):
        returns = list()
        returns.append(self.reward_buffer[-1])

        # Calculate return of each state
        self.reward_buffer.reverse()
        for idx in range(1, len(self.reward_buffer)):
            returns.append(self.reward_buffer[idx] +
                           self.gamma * returns[-1])
        returns.reverse()

        # Calculate advantage
        returns = T.from_numpy(np.asarray(returns)).float()
        state_value_buffer = T.from_numpy(
            np.asarray(self.state_value_buffer)).float()
        advantage = returns - state_value_buffer

        # Calculate log prob of actions
        action_prob_episode = T.stack(self.action_prob_buffer)
        log_action_prob_episode = T.log(action_prob_episode)

        # Calculate entropy
        action_probs_episode = T.stack(self.action_probs_buffer)
        entropy = (-action_probs_episode *
                   T.log(action_probs_episode)).sum(1).mean()

        # loss
        action_loss = ((-log_action_prob_episode * advantage) +
                       self.beta_entropy * entropy).mean()
        value_loss = advantage.pow(2).mean()
        total_loss = action_loss + value_loss

        # Optimize A2C model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train(self, start_state, env):
        state = start_state
        self.total_reward = 0
        self._reset_buffers()

        # env.set_initial_state(self.initial_state)
        for local_move_step in range(self.num_local_move):
            action = self._select_action_and_record_state_value(state)
            next_state, reward = env.step(
                state=state, position=action, flip=True)

            print(
                f"Local move iteration {local_move_step}, reward obtained {reward.item()}")
            self.state_buffer.append(state)
            self.action_buffer.append(action)
            self.reward_buffer.append(reward)

            state = next_state

        self._optimize_model()
