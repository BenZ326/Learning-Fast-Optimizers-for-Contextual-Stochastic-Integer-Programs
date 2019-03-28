from Model_KS import KS_MIP
from instance import Instance_KS

# create a knapsack instance
ks=Instance_KS()
# get items values
value=ks.get_values()
#sample a scenario and return a weight vecotr
weight=ks.sample_scenarios()
#get the capacity of the knapsack
capacity=ks.get_C()

#-------------------------------------------------finish creating an instance
#model a mip for knapsack problem
# need two arguements:
# the first argument is the instance
# the second argument is the sampled weight
ks_mip = KS_MIP(ks,weight)
# solve the problem, it will return the status of the solver, it can be "optimal" , "unbounded", "infeasible"
status = ks_mip.solve()
# get the optimal objective
obj = ks_mip.query_opt_obj()
# get the optimal solution
X=ks_mip.query_solution()

# get the slack variable
slack = ks_mip.query_slack()

print("the objective value is {}".format(obj),"the optimal solution is {}".format(X), "the slack variable is {}".format(slack))



