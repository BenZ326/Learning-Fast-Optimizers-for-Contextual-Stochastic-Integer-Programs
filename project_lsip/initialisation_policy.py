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

            p_val[i+1] = p_val[i] * \
                T.pow(p_dist_1[i], solution[i][0]) * \
                T.pow(1 - p_dist_1[i], 1-solution[i][0])
            a[i+1] = a[i] + self.W[:, self.dim_context + i] * solution[i][0]

        solution = T.stack(solution)
        return solution, T.log(p_val[i-1])

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


class RNNInitialisationPolicy(nn.Module):
    pass


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
