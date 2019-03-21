import torch as T
import torch.nn as nn
import numpy as np


class BinaryNADE(nn.Module):
    """
    Binary NADE module to learn a probability distribution over solution
    given the context vector i.e. P(x | c)
    """

    def __init__(self, dim_sol, dim_context, dim_hidden):
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

    def forward(self, context, solution):
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
        assert solution.shape[0] == self.dim_sol, "Solution dimension mismatch"

        a = self.c + T.matmul(self.W[:, : self.dim_context], context)
        p_val = 1
        p_dist = T.zeros(self.dim_sol)
        for i in range(self.dim_sol):
            h = T.sigmoid(a)
            p_dist[i] = T.sigmoid(self.b[i] + T.matmul(self.V[i], h)).view()
            p_val = p_val * (T.pow(p_dist[i], solution[i]) +
                             T.pow(1-p_dist[0], 1-solution[i]))
            a = a + self.W[:, self.dim_context + i] * solution[i]

        return p_val, p_dist


class Baseline(nn.Module):
    """
    A feedforward neural network to estimate the average reward given context
    i.e. b_w(c)
    """

    def __init__(self, dim_context, dim_hidden):
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.fc1 = nn.Linear(self.dim_context, self.dim_hidden)
        self.fc2 = nn.Linear(self.dim_hidden, 1)

    def forward(context):
        return self.fc2(T.ReLU(self.fc1(context)))


class InitializationPolicy(BinaryNADE):

    def act(self):
        pass


def REINFORCE():
    pass
