# An abstract class of Env
from Model_KS import KS_MIP
import numpy as np
from base_line_alg import extensive
import numpy.random as rd
import copy

from state import state

Number_Sampled_Scenarios = 40


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

    def __init__(self, args, instance, N_w, TIME_LIMIT=20):
        Env.__init__(self)
        self.args = args
        self.instance = instance
        self.N_w = N_w
        self._action_ex = None
        # Time limit for the SCIP solver
        self.TIME_LIMIT = TIME_LIMIT

    def extensive_form(self):
        ex_model = extensive(self.instance, self.N_w)
        ex_model.solve(self.TIME_LIMIT)
        self._action_ex = ex_model.solution
        return ex_model.solution, ex_model.opt_obj, ex_model.gap, ex_model.best_sol_list

    def step(self, state=None, sol=None, pos=None, flip=False):
        """
        Arguments
        ---------
        action: a solution popped out by NADE

        Returns
        -------
        return value: reward
        """
        if flip:
            new_sol = copy.deepcopy(state.get_sol())
            if new_sol[pos] == 0:
                new_sol[pos] = 1
            else:
                new_sol[pos] = 0
            new_obj, scenarios_vec = self.evaluate(new_sol, True)
            assert (state.get_obj() == None)
            reward = new_obj - state.get_obj()
            refined_scenarios = self.refine_scenarios(scenarios_vec)
            state.update((new_sol, new_obj), refined_scenarios)
            return reward, state.get_state()
        else:
            obj_value, scenarios_vec = self.evaluate(sol, True)
            refined_scenarios = self.refine_scenarios(scenarios_vec)

            state_init = state(args, tuple([sol, obj_value]),
                               self.instance.get_context())
            state_init.update((sol, obj_value), refined_scenarios)

            return np.array([obj_value]).reshape(-1), state.get_state()

    def evaluate(self, sol, memory=False):
        """
        if memory is true, it means we need to store the scenarios and sampled scenarios
        """

        scenario_vec = []
        f_x = sol @ self.instance.get_values()
        f_y = 0  # the expected cost of second stage
        for scenario in range(self.N_w):
            model = None
            while True:
                weights = self.instance.sample_scenarios()
                model = KS_MIP(self.instance, sol, weights)
                solver_state = model.solve()
                if solver_state == "optimal":
                    if memory:
                        tmp_vec = list(model.solution)
                        tmp_vec.append(model.slack)
                        scenario_vec.append(tmp_vec)
                    break
            f_y += model.query_opt_obj() / self.N_w
        return f_x + f_y, scenario_vec

    def refine_scenarios(self, scenarios_vec):
        """
        To make a difference in the function name, we call refine_scenarios as the function to uniformly sample a subset of scenarios
        """

        refined = []
        length_vec = len(scenarios_vec)
        while len(refined) <= Number_Sampled_Scenarios:
            idx = rd.randint(0, length_vec)
            if idx in refined:
                continue
            refined.append(idx)
        res = []
        for idx in refined:
            res.extend(scenarios_vec[idx])
        return res
