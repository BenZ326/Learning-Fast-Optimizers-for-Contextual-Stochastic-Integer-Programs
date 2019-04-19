import numpy as np
import environment


class State:
    """
    A representation of the state for the local move policy
    """

    def __init__(self, args, solution, obj_value, context, latest_scenario_solution_and_slack):
        """
        Intialize state

        Arguments
        ---------

        args : dict
            Dictionary containing command line arguments
        solution : list
            First-stage solution
        obj_value : float
            Objective value of the above solution
        context : list
            Context vector containing the description of an instance
        """
        self.args = args
        self._context = context

        # Store the solutions to scenarios and their corresponding slack.
        # We are assuming that this state is for KS and hence it will have
        # only one constraint. Hence, only one slack per scenario.
        self._scenario_solution_and_slack = np.zeros(
            args.window_size * (args.num_of_scenarios_in_state * (args.dim_problem + 1)))

        self.update(solution, obj_value, latest_scenario_solution_and_slack)

    def update(self, solution, obj_value, latest_scenario_solution_and_slack):
        """
        Update the state for the local move policy after flipping
        a bit

        solution : list
            the new first-stage solution of popped out by the local move policy
        obj_value : list
            the objective value of the new solution
        latest_scenario_solution_and_slack : list
            the aggregated information retrieved from the environment at the current iteration
        """
        self._solution = solution
        self._obj_value = obj_value

        # Extend the solutions for the last time windown and crop the solutions of the first
        # time window
        self._scenario_solution_and_slack = np.append(self._scenario_solution_and_slack,
                                                      latest_scenario_solution_and_slack, axis=0)
        self._scenario_solution_and_slack = self._scenario_solution_and_slack[self.args.num_of_scenarios_in_state*(
            self.args.dim_problem+1):]

    # Access private variable solution
    def get_solution(self):
        return self._solution

    # Access private variable obj_value
    def get_obj_value(self):
        return self._obj_value

    # Access private variable scenario_solution_and_slack
    def get_scenario_solution_and_slack(self):
        return self._scenario_solution_and_slack

    # Access private variable context
    def get_context(self):
        return self._context

    # Get the representation of the state in a vector form
    def get_representation(self):
        """
        Aggregate the information in state object to generate a representation
        """

        representation = list()
        representation.extend(self.get_solution())
        representation.extend(self.get_context())
        representation.extend(self.get_scenario_solution_and_slack())

        return np.asarray(representation)
