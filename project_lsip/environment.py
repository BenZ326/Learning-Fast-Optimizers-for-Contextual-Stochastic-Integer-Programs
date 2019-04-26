import numpy as np
import numpy.random as rd
import copy

from State import State
from Model_KS import KS_MIP
from base_line_alg import extensive


class Environment:
    """
    Abstract class for environment. Each specific environment 
    must inherit this class and define the methods
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass


class EnvKnapsack(Environment):
    """
    Environment for the Knapsack problem
    """

    def __init__(self, args, instance, TIME_LIMIT=20):
        """
        Arguments
        ---------
        instance: a problem instance

        Returns
        -------
        """

        Environment().__init__()
        self.args = args
        self.instance = instance
        self._action_ex = None
        # Time limit for the SCIP solver
        self.TIME_LIMIT = TIME_LIMIT

    def extensive_form(self):
        ex_model = extensive(
            self.instance, self.args.num_of_scenarios_for_expectation)
        ex_model.solve(self.TIME_LIMIT)
        self._action_ex = ex_model.solution
        return ex_model.solution, ex_model.opt_obj, ex_model.gap, ex_model.best_sol_list

    def step(self, solution=None, obj_value=None, state=None, position=None, flip=False):
        """
        Evaluate an action taken by the agent

        Arguments
        ---------
        solution :
            #TODO
        obj_value :
            #TODO
        state : 
            #TODO
        position : 
            #TODO
        flip : 
            #TODO

        Returns
        -------
        return value: reward
        """
        # Called from local move policy
        if flip:
            # Generate new solution
            new_solution = copy.deepcopy(state.get_solution())
            # Check whether we are required to make an update
            if position < self.args.dim_problem:
                if new_solution[position] == 0:
                    new_solution[position] = 1
                else:
                    new_solution[position] = 0

            # Evaluate the new solution and to get its objective value and
            # scenario solutions & slack
            new_obj_value, all_scenarios_solution_and_slack = self.evaluate(
                new_solution, True)

            # Reward for the local move policy, which is nothing but the difference in
            # the objective value corresponding to the new and old solutions
            reward = new_obj_value - state.get_obj_value()

            # TRICK: Randomly sample a subset of scenarios to form the new state
            # We cannot consider all the scenarios as it will blow up the state size
            scenarios_for_state_definition = self.sample_scenarios_for_state_definition(
                all_scenarios_solution_and_slack)
            state.update(new_solution, new_obj_value,
                         scenarios_for_state_definition)

            return state, reward

        # Called from Initialisation Policy
        else:
            obj_value, scenarios_vec = self.evaluate(solution, memory=True)
            scenarios_for_state_definition = self.sample_scenarios_for_state_definition(
                scenarios_vec)

            state = State(self.args, solution, obj_value,
                          self.instance.get_context(), scenarios_for_state_definition)

            return state, np.array([obj_value]).reshape(-1)[0]

    def evaluate(self, sol, memory=False):
        """
        if memory is true, it means we need to store the scenarios and sampled scenarios
        """

        scenario_vec = []
        f_x = sol @ self.instance.get_values()
        f_y = 0  # the expected cost of second stage
        for scenario in range(self.args.num_of_scenarios_for_expectation):
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
            f_y += model.query_opt_obj() / self.args.num_of_scenarios_for_expectation
        return f_x + f_y, scenario_vec

    def sample_scenarios_for_state_definition(self, scenarios_vec):
        """
        Select a subset of scenarios among all the scenarios considered to evaluate the expectation
        """

        refined = []
        length_vec = len(scenarios_vec)
        while len(refined) < self.args.num_of_scenarios_in_state:
            idx = rd.randint(0, length_vec)
            if idx in refined:
                continue
            refined.append(idx)
        res = []
        for idx in refined:
            res.extend(scenarios_vec[idx])
        return res
