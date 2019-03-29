import torch as T
import torch.nn as nn
import numpy as np


class BinaryNADE(nn.Module):
    """
    Binary NADE module to learn a probability distribution over solution
    given the context vector i.e. P(x | c)
    """

    def __init__(self, dim_sol, dim_context, dim_hidden):
        super(BinaryNADE, self).__init__()
        # Initialization const
        self.INIT = 0.01
        # Dimension of the variable vector
        self.dim_sol = dim_sol
        # Dimension of the context vector
        self.dim_context = dim_context
        # Dimension of the hidden_vector
        self.dim_hidden = dim_hidden
        # Parameter W (dim_hidden, dim_context + dim_input)
        self.W = nn.Parameter(T.Tensor(self.dim_hidden, self.dim_context +
                                       self.dim_input).uniform_(-self.INIT, self.INIT))
        # Parameter V (dim_sol x dim_hidden)
        self.U = nn.Parameter(
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
        solution = []

        a = self.c + T.matmul(self.W[:, : self.dim_context], context)
        p_val = 1
        p_dist_1 = T.zeros(self.dim_sol)
        p_dist_0 = T.zeros(self.dim_sol)
        for i in range(self.dim_sol):
            h = T.sigmoid(a)
            p_dist_1[i] = T.sigmoid(self.b[i] + T.matmul(self.V[i], h)).view()
            p_dist_0[i] = 1 - p_dist_1[i]
            # Sample the ith bit of the solution based on p_dist[i]
            solution.append(T.distributions.Bernoulli(
                [p_dist_1[i]]).sample()[0])
            p_val = p_val * (1 - T.pow(p_dist_0[i], solution[i]) +
                             1 - T.pow(p_dist_1[i], 1-solution[i]))
            a = a + self.W[:, self.dim_context + i] * solution[i]

        return solution, torch.log(p_val)


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

    def forward(context):
        return self.fc2(T.ReLU(self.fc1(context)))


class InitializationPolicy(BinaryNADE):

    def __init__(dim_sol, dim_context, dim_hidden):
        super(InitializationPolicy, self).__init__(
            dim_sol, dim_context, dim_hidden)

    def act(self, context):
        solution, log_prob = self.forward(context)
        return solution, log_prob


DIM_CONTEXT = 50
DIM_HIDDEN = 25
DIM_SOLUTION = 25

init_policy = InitializationPolicy(DIM_SOLUTION, DIM_CONTEXT, DIM_HIDDEN)
baseline = Baseline(DIM_CONTEXT, DIM_HIDDEN)
opt_init = T.optim.Adam(init_policy.parameters(), lr=1e-4)
opt_base = T.optim.Adam(baseline.parameters(), lr=1e-4)


def REINFORCE(context):
    solution, log_prob = init_policy.act(context)
    baseline_reward = baseline.forward(context)

    reward = env.step(solution)
    loss_init = -log_prob * reward
    loss_base = T.nn.MSELoss().(baseline_reward, reward)

    # Update the initialisation policy
    opt_init.zero_grad()
    loss_init.backward()
    opt_init.step()

    # Update the baseline
    opt_base.zero_grad()
    loss_base.backward()
    opt_base.step()


# def train(args):
#     # Generate environment
#     # Train for n steps
#     for epoch in range(args.EPOCHS):
#         instance = generate_instance()
#         REINFORCE(instance)


# def main():
#     # Extract args
#     train(args)


# if __name__ == "__main__":
#     main()
