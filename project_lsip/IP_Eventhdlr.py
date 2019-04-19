""""
To retrieve node information while a MIP is solved
"""
from pyscipopt import Model, Eventhdlr, quicksum, SCIP_EVENTTYPE
import numpy as np


class IP_Eventhdlr(Eventhdlr):
    """"
    collect data for IP event
    """

    def __init__(self):
        super(IP_Eventhdlr, self).__init__()
        self.IPSol = []

    def collect_node_info(self):
        best_sol = self.model.getBestSol()
        best_obj = self.model.getSolObjVal(best_sol)
        current_time = self.model.getSolvingTime()
        self.IPSol.append(tuple([best_obj, current_time]))

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND:
            self.collect_node_info()

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
