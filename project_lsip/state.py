import numpy as np
import  environment

Number_Sampled_Scenarios = 40
Time_Step = 5
Problem_Sizes = 10


class state:

    def __init__(self, init_sol, context_vec):
        self._sol = tuple([init_sol,None])
        self._aggregation = np.zeros(Time_Step * Number_Sampled_Scenarios*(Problem_Sizes+1))
        self._context_vector = context_vec

    def update(self, sol, new_info):
        new_info.extend(self._aggregation)
        self._aggregation = new_info
        self._aggregation = self._aggregation[:Time_Step*Number_Sampled_Scenarios*(Problem_Sizes+1)]
        self._sol = sol
    def get_sol(self):
        return self._sol[0]
    def get_obj(self):
        return self._sol[1]
    def get_aggregation(self):
        return self._aggregation
    def get_context(self):
        return self._context_vector

