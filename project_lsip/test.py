from Model_KS import KS_MIP
from instance import Instance_KS
from instance import instance_generator
import numpy as np
from Env import Env_KS

PROBLEM = "knapsack"

# create a knapsack instance
generator = instance_generator(PROBLEM)
ks=generator.generate_instance()
# get items values
value=ks.get_values()
x_star=np.zeros(len(value))
#sample a scenario and return a weight vecotr
weight=ks.sample_scenarios()
#get the capacity of the knapsack
capacity=ks.get_C()

#-------------------------------------------------finish creating an instance
#model a mip for knapsack problem
# need two arguements:
# the first argument is the instance
# the second argument is the sampled weight
ks_mip = KS_MIP(ks,x_star,weight)
# solve the problem, it will return the status of the solver, it can be "optimal" , "unbounded", "infeasible"
status = ks_mip.solve()
# get the optimal objective
obj = ks_mip.query_opt_obj()
# get the optimal solution
X=ks_mip.query_solution()

# get the slack variable
slack = ks_mip.query_slack()
print("the objective value is {}".format(obj),"the optimal solution is {}".format(X), "the slack variable is {}".format(slack))
print("context_vector is {}".format(ks.get_context()))

env = Env_KS(ks,100)
reward = env.step(x_star)
print("the reward of solution {}".format(x_star)," is {}".format(reward))
