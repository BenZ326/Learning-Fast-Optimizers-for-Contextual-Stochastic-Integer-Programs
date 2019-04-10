import math
import random
from pyscipopt import Model, quicksum
import numpy as np
#######################################################################
# Authors: Xiangyi Zhang, Rahul Patel
# The Module basically provides the class of a post-modeled problem
# The user is able to:
# 1) model a deterministic problem
# 2) solve the problem
# 3) access to the solver info, e.g. the status of a solver, computation time, node info ...
# 4) access to info on the optimal solutions
#######################################################################




"An abstract class"
class MIPModel:
    def __init__(self):
        pass
    """
    use pyscip to model the problem at hand
    """
    def _model(self):
        pass
    """
    call the optimize method by pyscip
    """
    def _optimize(self):
        pass

    def _extract_solver_info(self):
        pass
    """
    get optimal solution and the corresponding objective value
    """
    def _get_optimal(self):
        pass


"""
Knapsack problem modeler:
"""


# In the paper, precisely speaking, they are modeling a standard knapsack problem
#  Notations:
# Decision variable:
# x_{i}=1 if item i is put into the knapsack
# s: the slack variable for the capacity constraint
# maximize x_{i}*v_{i} for i=1...n
# s.t.
#     w_{i}*x_{i} + s == c       respect the capacity
#     x_{i} is binary
# -----------------------------model the deterministic knapsack problem

class KS_MIP(MIPModel):

    """
    KS: the knapsack instance

    fss: first_stage_solution
    scenario: the sampled scenario to be modeled.
    model the MIP

    """
    def __init__(self,ks,fss,sampled_weights):
        MIPModel.__init__(self)
        self.value = ks.get_values()
        self.penalty = ks.get_penalty()
        self.weight = sampled_weights
        self.capacity = ks.get_C()
        assert(len(fss) == len(self.value))
        self.x_star = fss

        self.model = Model("knap_sasck")
        self.solution = None
        self.slack = None
        self.opt_obj = None

    def solve(self):
        # Create variables
        y = {}
        # auxiliary variables
        ax = {}
        for i in range(len(self.value)):
            y[i] = self.model.addVar(vtype="B", name="y(%s)" % i)
            ax[i] = self.model.addVar(vtype="B", name="ax(%s)" % i)
        slack = self.model.addVar(vtype="C", name="slack")
        # Add constraints
        self.model.addCons(quicksum(self.weight[j] * y[j] for j in range(len(self.weight))) + slack == self.capacity,
                           name="capacity constraint")
        for j in range(len(self.value)):
            self.model.addCons(ax[j] >= self.x_star[j] - y[j])
        # impose ax[i] = max (y-x,0) api does not support min, max function
        self.model.setObjective(
            quicksum(self.value[j] * y[j] + ax[j] * (-1 * self.penalty) for j in range(len(self.value))), "maximize")
        self.model.data = y, slack
        self.model.hideOutput()             #silent the output

        self.model.optimize()
        status = self.model.getStatus()
        if status == "unbounded" or status == "infeasible":
            return status

        y, slack = self.model.data
        Y = np.zeros(len(self.weight))
        for i in range(len(self.weight)):
            Y[i] = self.model.getVal(y[i])
        self.solution, self.slack, self.opt_obj = Y, self.model.getVal(slack), self.model.getObjVal()


        return status

    def query_solution(self):
        return self.solution
    def query_slack(self):
        return self.slack
    def query_opt_obj(self):
        return self.opt_obj







