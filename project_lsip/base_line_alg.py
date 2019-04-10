import numpy as np
from pyscipopt import Model, quicksum
from instance import Instance_KS

import copy
class extensive:
    def __init__(self,instance,n_w):
        self._instance= copy.deepcopy(instance)
        self._n_w = n_w
        self.solution = None
        self.opt_obj = None
        self.gap = None
    def solve(self,time_limit):
        ex_model = Model("extensive model")
        x = {}
        y = {}
        values = self._instance.get_values()
        for i in range(len(values)):
            x[i] = ex_model.addVar(vtype = "B",name="x(%s)" % i)

        for s in range(self._n_w):
            for i in range(len(values)):
                y[i,s] = ex_model.addVar(vtype = "B", name = "y{},{}".format(i,s))
                ex_model.addCons(x[i] >= y[i, s],"x,y{},sc:{}".format(i,s))
        for s in range(self._n_w):
            w = self._instance.sample_scenarios()
            ex_model.addCons(quicksum(w[i]*(x[i]-y[i,s]) for i in range(len(values)))<=self._instance.get_C(),"respect capacity for scenario {}".format(s))
        ex_model.setObjective(quicksum(x[i]*values[i] for i in range(len(values)))-(1/self._n_w)
                              *quicksum(quicksum(y[i,s]*self._instance.get_penalty() for i in range(len(values))) for s in range(self._n_w)),
                              "maximize")
        ex_model.data = x, y
        ex_model.hideOutput()
        ex_model.setRealParam('limits/time',time_limit)
        ex_model.optimize()
        self.gap = ex_model.getGap()
        status = ex_model.getStatus()
        if status == "unbounded" or status == "infeasible":
            return status
        x, y = ex_model.data
        self.solution = np.zeros(len(values))
        for i in range(len(values)):
            self.solution[i] = ex_model.getVal(x[i])
        self.opt_obj = ex_model.getObjVal()
