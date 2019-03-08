import numpy as np
import random as rd

#######################################################################
# Authors: Xiangyi Zhang, Rahul Patel                                 #
#######################################################################
"""
The script is to generate random instances of stochastic knapsack problems where uncertainty comes from values of items.
Configurations of the instances are as follows.
1.The capacity of the knapsack follows a discrete uniform distribution, say [100, 150, 200, 250, 300];
2.Any item's value follows a uniform distribution [v_min, v_max], where v_min and v_max respectively sampled from a normal distribution;
3.Any item's weight is given by sampling from a distribution governed by parameters $h$, the percentage of capacity, $\delta$ the standard 
variation of a normal distribution, as well as the number of items |N|. Basically it can be written as:
w = C/(h*|N|) + e and e follows N(0,$\delta$^2);
"""

N = 30 # number of items
Support_C = np.array([100,150,200,250,300]) # support of the uniform distribution for capacity
Support_H = np.array([0.2, 0.4, 0.6,0.8])   # support of the uniform distribution for percentage of items to be packed in expect
idx_c = rd.randint(0,4)
idx_h = rd.randint(0,3)
Penalty = rd.randint(3,8)
C = Support_C[idx_c]
H = Support_H[idx_h]
delta = 0.01*C
while(True):
    v_min = rd.gauss(10,4)
    v_max = rd.gauss(25,4)
    if v_max>v_min:
        break

# generate items
# To avoid numerical issues in solvers we round the values and weights to the nearest integers
V=np.round(np.random.uniform(v_min,v_max,N))


print("The percentage of items to be packed in expect is ",H)
print("\nThe Knapsack problem is as follows: \n Given a knapsack with capacity {} ".format(C)+ 
      "there are {} items, with uncertain weights vector follows a gaussian distribution N({},{})and a values vector {}. Determine which items should be packed in order".format(N,C/(H*N),delta**2,V)+
      " to maximize the total values of items in the knapsack. ")



def sample_scenarios():
    W = np.round(np.random.normal(C/(H*N),delta**2,N)) # weights are not known in advance
    return W


