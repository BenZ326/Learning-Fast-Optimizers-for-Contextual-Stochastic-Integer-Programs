import math
import random
from pyscipopt import Model

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
class KS_MIP(MIPModel):

    def __init__(self):
        MIPModel.__init__()





