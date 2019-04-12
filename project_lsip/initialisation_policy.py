import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.bernoulli import Bernoulli


class NADEInitializationPolicy(nn.Module):
    """
    Binary NADE module to learn a probability distribution over solution
    given the context vector i.e. P(x | c)
    """

    def __init__(self, dim_problem, dim_context, dim_hidden):
        super(NADEInitializationPolicy, self).__init__()
        # Initialization const
        self.INIT = 0.01
        # Dimension of the variable vector
        self.dim_problem = dim_problem
        # Dimension of the context vector
        self.dim_context = dim_context
        # Dimension of the hidden_vector
        self.dim_hidden = dim_hidden
        # Parameter W (dim_hidden, dim_context + dim_problem)
        self.W = nn.Parameter(T.Tensor(self.dim_hidden, self.dim_context +
                                       self.dim_problem).uniform_(-self.INIT, self.INIT))
        # Parameter V (dim_problem x dim_hidden)
        self.V = nn.Parameter(
            T.Tensor(self.dim_problem, self.dim_hidden).uniform_(-self.INIT, self.INIT))
        # Parameter b (dim_problem x 1)
        self.b = nn.Parameter(
            T.Tensor(self.dim_problem).uniform_(-self.INIT, self.INIT))
        # Parameter c (dim_hidden x 1)
        self.c = nn.Parameter(
            T.Tensor(self.dim_hidden).uniform_(-self.INIT, self.INIT))

    def forward(self, context):
        """
        Forward pass of NADE

        Parameters
        ----------
        solution : binary vector
            The tensor required to regress on by NADE

        Return
        ------
        p_val : float
            Joint probability of the tensor
        p_dist : float vector
            Probability distribution along the dimensions of x conditioned on
            context
        """
        assert context.shape[0] == self.dim_context, "Context dimension mismatch"
        context = T.from_numpy(context).float()

        solution = []
        h = {}
        a = {}
        p_val = {}
        p_dist_1 = {}

        a[0] = self.c + T.matmul(self.W[:, : self.dim_context], context)
        p_val[0] = 1
        #  = T.zeros(self.dim_problem)

        for i in range(self.dim_problem):
            h[i] = T.sigmoid(a[i])
            p_dist_1[i] = T.sigmoid(
                self.b[i] + T.matmul(self.V[i], h[i]))

            # Sample the ith bit of the solution based on p_dist[i]
            solution.append(T.distributions.Bernoulli(
                p_dist_1[i].clone().detach().unsqueeze(0)).sample())
            # Update the joint probability
            p_val[i+1] = p_val[i] * \
                T.pow(p_dist_1[i], solution[i][0]) * \
                T.pow(1 - p_dist_1[i], 1-solution[i][0])
            a[i+1] = a[i] + self.W[:, self.dim_context + i] * solution[i][0]

        solution = T.stack(solution)
        return solution, T.log(p_val[i+1])

    def REINFORCE(self, opt_init, env, context, baseline_reward=None, use_baseline=False):
        solution, log_prob = self.forward(context)

        # Get reward from the environment and calculate loss
        reward = T.from_numpy(env.step(solution.numpy().reshape(-1))).float()
        if use_baseline:
            loss_init = -log_prob * (reward - baseline_reward.item())
        else:
            loss_init = -log_prob * reward
        print("Reward: {}".format(reward))

        # Update the initialisation policy
        opt_init.zero_grad()
        loss_init.backward()
        opt_init.step()

        return reward, loss_init


class NNInitialisationPolicy(nn.Module):
    def __init__(self, dim_problem, dim_context, dim_hidden):
        super(NNInitialisationPolicy, self).__init__()
        # Initialization const
        self.INIT = 0.01
        # Dimension of the variable vector
        self.dim_problem = dim_problem
        # Dimension of the context vector
        self.dim_context = dim_context
        # Dimension of the hidden_vector
        self.dim_hidden = dim_hidden
        self.linear_1 = nn.Linear(self.dim_context, self.dim_hidden)
        self.linear_2 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.output = nn.Linear(self.dim_hidden, self.dim_problem)

    def forward(self, context):
        context = T.from_numpy(context).float()

        x = F.relu(self.linear_1(context))
        x = F.relu(self.linear_2(x))
        solution_probs = T.sigmoid(self.output(x))
        return solution_probs

    def REINFORCE(self, opt_init, env, context, baseline_reward=None, use_baseline=False):
        joint = {}
        print("Inside reinforce")
        # Sample solution
        solution_probs = self.forward(context)
        m = Bernoulli(solution_probs)
        solution = m.sample().detach()

        # Find the joint probability
        joint[0] = T.ones(1)
        for i in range(len(solution)):
            if solution[i].item() == 0:
                joint[i+1] = joint[i] * (1-solution_probs[i])
            elif solution[i].item() == 1:
                joint[i+1] = joint[i] * solution_probs[i]

        # Get reward from the environment
        reward = T.from_numpy(env.step(solution.numpy().reshape(-1))).float()
        # i+1th element will hold the joint probability of the sampled
        # solution
        if use_baseline:
            loss_init = -T.log(joint[i+1]) * \
                (reward - baseline_reward.clone().detach())
        else:
            loss_init = -T.log(joint[i+1]) * reward

        print("Reward: {}".format(reward))

        # Update the initialisation policy
        opt_init.zero_grad()
        loss_init.backward()
        opt_init.step()

        return reward, loss_init


class LSTMInitialisationPolicy(nn.Module):
    def __init__(self, dim_problem, dim_context, dim_hidden):
        super(LSTMInitialisationPolicy, self).__init__()
        # Initialization const
        self.INIT = 0.01
        # Dimension of the variable vector
        self.dim_problem = dim_problem
        # Dimension of the context vector
        self.dim_context = dim_context
        # Dimension of the hidden_vector
        self.dim_hidden = dim_hidden
        self.dim_lstm_in = 1

        self.linear_1 = nn.Linear(self.dim_context, self.dim_lstm_in)

        self.W_f = nn.Linear(
            self.dim_hidden + self.dim_lstm_in, self.dim_hidden)

        self.W_i = nn.Linear(self.dim_hidden +
                             self.dim_lstm_in, self.dim_hidden)

        self.W_c = nn.Linear(self.dim_hidden +
                             self.dim_lstm_in, self.dim_hidden)

        self.W_o = nn.Linear(self.dim_hidden +
                             self.dim_lstm_in, self.dim_lstm_in)

    def forward(self, context, h=None, c=None):
        context = T.from_numpy(context).float()
        # Store the graph of LSTM in the forward pass
        _f = {}
        _i = {}
        _c = {}
        __c = {}
        _h = {}
        _o = {}
        # Store input and joint probabilities
        x = {}
        p_val = {}
        p_val[0] = 1
        solution = []

        if h == None and c == None:
            _h[-1] = T.Tensor(self.dim_hidden)
            _c[-1] = T.zeros(self.dim_hidden)

            # Convert context to 1x1
        x[0] = self.linear_1(context)
        for t in range(self.dim_problem):
            concat_input = T.cat((_h[t-1], x[t]), 0)

            # Forget
            _f[t] = T.sigmoid(self.W_f(concat_input))

            # Input
            _i[t] = T.sigmoid(self.W_i(concat_input))

            # New information to choose from
            __c[t] = T.tanh(self.W_c(concat_input))

            # New cell state. Forget from the past state and add from
            # the current state to create the new cell state
            _c[t] = _f[t] * _c[t-1] + _i[t] * __c[t]

            # Output
            _o[t] = T.sigmoid(self.W_o(concat_input))

            # Compute hidden state for the next time step
            _h[t] = _o[t] * T.tanh(_c[t])

            # Sample the t^th bit of the solution based on _o[t]
            solution.append(T.distributions.Bernoulli(
                _o[t].clone().detach().unsqueeze(0)).sample())
            # Building input on the fly
            x[t+1] = solution[-1][0].clone().detach()

            # Update the joint probability
            p_val[t+1] = p_val[t] * \
                T.pow(_o[t], solution[t][0]) * \
                T.pow(1 - _o[t], 1-solution[t][0])

        solution = T.stack(solution)
        return solution, T.log(p_val[t+1])

    def REINFORCE(self, opt_init, env, context, baseline_reward=None, use_baseline=False):
        solution, log_prob = self.forward(context)

        # Get reward from the environment and calculate loss
        reward = T.from_numpy(env.step(solution.numpy().reshape(-1))).float()
        if use_baseline:
            loss_init = -log_prob * (reward - baseline_reward.item())
        else:
            loss_init = -log_prob * reward
        print("Reward: {}".format(reward))

        # Update the initialisation policy
        opt_init.zero_grad()
        loss_init.backward()
        opt_init.step()

        return reward, loss_init


class Baseline(nn.Module):
    """
    A feedforward neural network to estimate the average reward given context
    i.e. b_w(c)
    """

    def __init__(self, dim_context, dim_hidden):
        super(Baseline, self).__init__()
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.fc1 = nn.Linear(self.dim_context, self.dim_hidden)
        self.fc2 = nn.Linear(self.dim_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, context):
        context = T.from_numpy(context).float()
        out = self.relu(self.fc1(context))
        out = self.fc2(out)
        return out


# if __name__ == "__main__":
#     pass
