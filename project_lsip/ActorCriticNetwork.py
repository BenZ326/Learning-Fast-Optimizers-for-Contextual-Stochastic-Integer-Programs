import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    def __init__(self, dim_context, dim_problem, window_size, num_of_scenarios_in_state):
        super(ActorCriticNetwork, self).__init__()
        # Initialize network input and output dimensions
        self.dim_state = dim_problem + dim_context + window_size * \
            (num_of_scenarios_in_state * (dim_problem + 1))
        self.dim_hidden_1 = int(self.dim_state / 3)
        self.dim_hidden_2 = int(self.dim_state / 3)
        self.dim_action = dim_problem + 1
        self.dim_state_value = 1
        self.softmax = nn.Softmax(dim=0)

        self._initialize_network()

    def _initialize_network(self):
        self.linear_1 = nn.Linear(self.dim_state, self.dim_hidden_1)
        self.linear_2 = nn.Linear(self.dim_hidden_1, self.dim_hidden_2)
        self.linear_3 = nn.Linear(self.dim_hidden_2, self.dim_state_value)
        self.linear_4 = nn.Linear(self.dim_hidden_2, self.dim_action)

    def get_action_probs_and_state_value(self, state):
        x = F.relu(self.linear_1(state))
        x = F.relu(self.linear_2(x))

        state_value = self.linear_3(x)
        action_probs = self.softmax(self.linear_4(x))

        return action_probs, state_value
