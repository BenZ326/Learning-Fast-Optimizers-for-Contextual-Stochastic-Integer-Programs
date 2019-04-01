import torch as T
import torch.nn as nn
import numpy as np


class InitializationPolicy(nn.Module):
    """
    Binary NADE module to learn a probability distribution over solution
    given the context vector i.e. P(x | c)
    """

    def __init__(self, dim_problem, dim_context, dim_hidden):
        super(InitializationPolicy, self).__init__()
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

            p_val[i+1] = p_val[i] * \
                T.pow(p_dist_1[i], solution[i][0]) * \
                T.pow(1 - p_dist_1[i], 1-solution[i][0])
            a[i+1] = a[i] + self.W[:, self.dim_context + i] * solution[i][0]

        solution = T.stack(solution)
        return solution, T.log(p_val[i-1])

    def REINFORCE(self, opt_init, env, context):
        solution, log_prob = self.forward(context)
        # baseline_reward = baseline.forward(context)
        reward = T.from_numpy(env.step(solution.numpy().reshape(-1))).float()

        loss_init = -log_prob * reward
        print("Reward: {}\t Loss: {}".format(reward, loss_init))
        # loss_base = T.nn.MSELoss().(baseline_reward, reward)

        # Update the initialisation policy
        opt_init.zero_grad()
        loss_init.backward()
        opt_init.step()

        # Update the baseline
        # opt_base.zero_grad()
        # loss_base.backward()
        # opt_base.step()

        solution, log_prob = self.forward(context)
        # baseline_reward = baseline.forward(context)
        reward_ = T.from_numpy(env.step(solution.numpy().reshape(-1))).float()
        # baseline = baseline_net.forward(context)
        # print(log_prob, baseline, reward_)
        # loss_init_ = -log_prob * (reward_ - baseline.clone().detach().float())
        loss_init_ = -log_prob * reward_
        print("Reward: {}\t Loss: {}".format(reward_, loss_init_))
        # baseline_output = loss_base(baseline, reward_)

        # Update the initialisation policy
        opt_init.zero_grad()
        loss_init_.backward()
        opt_init.step()

        # Update the baseline
        # opt_base.zero_grad()
        # baseline_output.backward()
        # opt_base.step()

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
