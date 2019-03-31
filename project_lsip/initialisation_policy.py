from Env import Env_KS
from instance import instance_generator
# from instance import Instance_KS

import torch as T
import torch.nn as nn
import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_epochs", type=int, default=1000)
    parser.add_argument("--dim_context", type=int, default=32)
    parser.add_argument("--dim_hidden", type=int, default=10)
    parser.add_argument("--dim_sol", type=int, default=30)
    args = parser.parse_args()
    return args


class InitializationPolicy(nn.Module):
    """
    Binary NADE module to learn a probability distribution over solution
    given the context vector i.e. P(x | c)
    """

    def __init__(self, dim_sol, dim_context, dim_hidden):
        super(InitializationPolicy, self).__init__()
        # Initialization const
        self.INIT = 0.01
        # Dimension of the variable vector
        self.dim_sol = dim_sol
        # Dimension of the context vector
        self.dim_context = dim_context
        # Dimension of the hidden_vector
        self.dim_hidden = dim_hidden
        # Parameter W (dim_hidden, dim_context + dim_sol)
        self.W = nn.Parameter(T.Tensor(self.dim_hidden, self.dim_context +
                                       self.dim_sol).uniform_(-self.INIT, self.INIT))
        # Parameter V (dim_sol x dim_hidden)
        self.V = nn.Parameter(
            T.Tensor(self.dim_sol, self.dim_hidden).uniform_(-self.INIT, self.INIT))
        # Parameter b (dim_sol x 1)
        self.b = nn.Parameter(
            T.Tensor(self.dim_sol).uniform_(-self.INIT, self.INIT))
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
        #  = T.zeros(self.dim_sol)
        for i in range(self.dim_sol):
            h[i] = T.sigmoid(a[i])
            p_dist_1[i] = T.sigmoid(
                self.b[i] + T.matmul(self.V[i], h[i]))

            # Sample the ith bit of the solution based on p_dist[i]
            solution.append(T.distributions.Bernoulli(
                p_dist_1[i].detach().unsqueeze(0)).sample())

            p_val[i+1] = p_val[i] * \
                T.pow(p_dist_1[i], solution[i][0]) * \
                T.pow(1 - p_dist_1[i], 1-solution[i][0])
            a[i+1] = a[i] + self.W[:, self.dim_context + i] * solution[i][0]

        solution = T.stack(solution)
        return solution, T.log(p_val[i-1])


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

    def forward(self, context):
        return self.fc2(T.ReLU(self.fc1(context)))


# class InitializationPolicy(BinaryNADE):

#     def __init__(self, dim_sol, dim_context, dim_hidden):
#         super(InitializationPolicy, self).__init__(
#             dim_sol, dim_context, dim_hidden)

#     def act(self, context):
#         print("Act....")
#         solution, log_prob = self.forward(context)
#         return solution, log_prob


DIM_CONTEXT = 32
DIM_HIDDEN = 10
DIM_SOLUTION = 30
KNAPSACK = "ks"


def REINFORCE(init_policy, opt_init, env, context):
    solution, log_prob = init_policy.forward(context)
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
    return reward, loss_init


def train(args):
    print("Inside Train...")
    reward = []
    loss_init = []
    generator = instance_generator(KNAPSACK)

    # InitializationPolicy
    init_policy = InitializationPolicy(
        args.dim_sol, args.dim_context, args.dim_hidden)
    opt_init = T.optim.Adam(init_policy.parameters(), lr=1e-4)

    # baseline = Baseline(args.dim_context, args.dim_hidden)
    # opt_base = T.optim.Adam(baseline.parameters(), lr=1e-4)

    # Train
    for epoch in range(1, args.init_epochs+1):
        print("Epoch : {}".format(epoch))
        # Generate instance
        instance = generator.generate_instance()
        context = instance.get_context()
        # Generate environment
        env = Env_KS(instance, 2000)
        # Learn using REINFORCE
        reward_, loss_init_ = REINFORCE(init_policy, opt_init, env, context)
        reward.append(reward_.item())
        loss_init.append(loss_init_.item())
        if epoch % 50 == 0:
            # Save the data file
            np.save("reward.npy", reward)
            np.save("loss_init.npy", loss_init)
            T.save(init_policy.state_dict(), "init_policy")

    np.save("reward.npy", reward)
    np.save("loss_init.npy", loss_init)
    T.save(init_policy.state_dict(), "init_policy")


if __name__ == "__main__":
    args = parse_args()
    train(args)
