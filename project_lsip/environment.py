# An abstract class of Env
from Model_KS import KS_MIP
import numpy as np
from base_line_alg import  extensive


class Env:
    def __init__(self):
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def step(self, action):
        pass

# Environment for two stage stochastic programming

#######################################################################
# Authors: Xiangyi Zhang, Rahul Patel                                 #
#######################################################################


class Env_KS(Env):
    """
    Arguments
    ---------        
    instance: a problem instance

    Returns
    -------
    N_w: the number of scenarios
    """

    def __init__(self, instance, N_w):
        Env.__init__(self)
        self.instance = instance
        self.N_w = N_w
        self._action_ex = None

    def extensive_form(self):
        ex_model = extensive(self.instance,self.N_w)
        ex_model.solve(200)
        self._action_ex = ex_model.solution
        return ex_model.solution, ex_model.opt_obj, ex_model.gap

    def step(self, action):
        """
        Arguments
        ---------
        action: a solution popped out by NADE

        Returns
        -------
        return value: reward
        """
        # get the objective value of first stage
        print("the action is {}".format(action))
        f_x = action@self.instance.get_values()
        f_y = 0         # the expected cost of second stage
        for scenario in range(self.N_w):
            model = None
            while True:
                weights = self.instance.sample_scenarios()
                model = KS_MIP(self.instance, action, weights)
                state = model.solve()
                if state == "optimal":
                    break
            f_y += model.query_opt_obj()/self.N_w
        return np.array([f_x + f_y]).reshape(-1)
