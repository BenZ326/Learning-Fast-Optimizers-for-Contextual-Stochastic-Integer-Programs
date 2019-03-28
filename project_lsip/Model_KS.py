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
    scenario: the sampled scenario to be modeled.
    model the MIP
    """
    def __init__(self,ks,sampled_weights):
        MIPModel.__init__(self)
        self.value = ks.get_values()
        self.weight = sampled_weights
        self.capacity = ks.get_C()
        self.model = Model("knap_sasck")
        self.solution = None
        self.slack = None
        self.opt_obj = None
    def solve(self):
        # Create variables
        x = {}
        for i in range(len(self.value)):
            x[i] = self.model.addVar(vtype="B", name="x(%s)" % i)
        slack = self.model.addVar(vtype="C", name="slack")
        # Add constraints
        self.model.addCons(quicksum(self.weight[j] * x[j] for j in range(len(self.weight))) + slack == self.capacity,
                    name="capacity constraint")
        self.model.setObjective(quicksum(self.value[j] * x[j] for j in range(len(self.value))), "maximize")
        self.model.data = x, slack
        self.model.optimize()
        status = self.model.getStatus()
        if status == "unbounded" or status == "infeasible":
            return status
        x, slack = self.model.data
        X = np.zeros(len(self.weight))
        for i in range(len(self.weight)):
            X[i] = self.model.getVal(x[i])
        self.solution,self.slack,self.opt_obj = X,self.model.getVal(slack),self.model.getObjVal()
        return status

    def query_solution(self):
        return self.solution
    def query_slack(self):
        return self.slack
    def query_opt_obj(self):
        return self.opt_obj







