import numpy as np
import random as rd
import copy
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
"""
The class has the following attributes:
1. self._N the number of items
2. self._C: the knapsack capacity
3. self._Penalty: the unit penalty of removing an item
4. self._delta: variance of the gaussian distribution followed by weights
5. self._H: the percentage of items to be packed in expect
6. self._V: the values of items
"""
class Instance_KS:
    def __init__(self, n=30, support_c=np.array([100,200,300]),
                 support_h = np.array([0.2, 0.4, 0.6]),
                 lb_p = 3, ub_p = 8, delta_ratio = 0.01,
                 m_v_min = 10, m_v_max = 40, var_v = 4):
        self._N = n
        self._Penalty = rd.randint(lb_p,ub_p)
        idx_c = rd.randint(0, len(support_c) - 1)
        idx_h = rd.randint(0, len(support_h) - 1)
        self._C = support_c[idx_c]
        self._H = support_h[idx_h]
        # when capacity is large and percentage of items to be packed is also very large,
        # then the mean value of weights would be small while its variance is large
        # there will be good chance to generate instances with negative weights, to avoid this happen
        # we set an upperbound of the variance
        upperbound_var = max(support_c)*delta_ratio
        self._delta = min(delta_ratio*self._C, np.sqrt(upperbound_var))
        v_min, v_max =0,0
        while (True):
            v_min = rd.gauss(m_v_min,var_v)
            v_max = rd.gauss(m_v_max,var_v)
            if v_min < v_max:
                break
        self._V = np.round(np.random.uniform(v_min,v_max,n))
        self._context_vector = np.insert(self._V,0,self._Penalty,axis = 0)
        self._context_vector = np.insert(self._context_vector,0,self._C, axis = 0)
        print("The instance will generate scenarios which follows Gaussian Distribution with {} as mean and {} as variance".format(self._C/(self._H*self._N),self._delta**2,self._N))
    def sample_scenarios(self):
        W = np.round(np.random.normal(self._C/(self._H*self._N),self._delta**2,self._N)) # weights are not known in advance
        return W
    def get_values(self):
        return self._V
    def get_penalty(self):
        return self._Penalty
    def get_C(self):
        return self._C
    def get_H(self):
        return self._H
    def get_context(self):
        return self._context_vector


"""
t = 0
KS = Instance_KS()
w = KS.sample_scenarios()
print(w,"\n", KS.get_values())

while (True):
    KS = Instance_KS()
    w = KS.sample_scenarios()
    if w.all()<=0:
        print("C = {}".format(KS.get_C()),"\n H = {}".format(KS.get_H()))
        break
    t += 1

print(t)
"""

""""
instance generator
argument1: the name of instance
"""
class instance_generator:
    def __init__(self,name):
        assert(name.lower()=="knapsack" or name.lower() == "facility location")
        self.name = name
    def generate_instance(self):
        if self.name.lower() == "knapsack":
            return Instance_KS()


