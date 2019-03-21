
import numpy as np
import random as rd
import copy

'''
 This code is for the chapter 4 of introduction to reinforcement learning
 The problem is to use policy evaluation, policy iteration to find the optimal policy for an agent to play grid world.

'''

Actions = {"0":np.array([0,-1]),            # left
           "1":np.array([-1,0]),            # up
           "2":np.array([1,0]),             # right
           "3":np.array([0,1])              # down
           }
Gamma = 0.01
Action_prob=0.25
# first part would be building the grid world.
class Grid:
    '''
    "size" is the size of the grid, int type
    "dist" is the probability distribution indicating the uncertainty of the environment, a list type where the first
    element is the probability of following the action, the second is the probability of randomly choosing an direction.
    '''
    def __init__(self, size, dist):
        self.size=size
        self.distribution = dist
        self._generate_grids()
        self.reward = -1
    '''
    based on the size, generate the grids and the indices for grids
    '''
    def _generate_grids(self):
        self.grids = np.zeros([self.size, self.size])
    '''
    decide a direction to go from the current location, there will be a chance to move another direction. 
    "direction": 1==>left, 2==>up, 3==>right, 4==>down, int 
    the index is : in the case of self.size = 5
    |(0,0) (0,1) (0,2) (0,3) (0,4)
    |(1,0) (1,1) (1,2) (1,3) (1,4)
    |(2,0) (2,1) (2,2) (2,3) (2,4)
    |(3,0) (3,1) (3,2) (3,3) (3,4)
    |(4,0) (4,1) (4,2) (4,3) (4,4)
    |____________________
    '''
class Agent:
    '''
    An agent has attributes:
        1) x_pos, y_pos
        2) parameters concerning the environment
        3) policy
        4) state-value

    '''

    def __init__(self, Env):
        self._Env = copy.deepcopy(Env)
        self._initial_pos()
        self._initial_policy()
        self._state_value = np.zeros([self._Env.size,self._Env.size])
        self._state_value[0][0]=10
        self._state_value[self._Env.size-1][self._Env.size-1]=10

    '''
    initialize the policy
    here we use dictionary as the data structure to represent a policy. key is the state, "(x_pos, y_pos)" : value is a
    list [, , , ,] where the elements are the probability of moving toward a direction. 
    '''
    def _initial_policy(self):
        self._policy = dict()
        for i in range(self._Env.size):
            for j in range(self._Env.size):
                self._policy[str((i,j))] = np.ones(4)*0.25

    '''
    Initialize the position
    '''
    def _initial_pos(self):
        assert (self._Env.size % 2 == 1)
        self._pos = np.zeros(2)
        self._pos[0] = int((self._Env.size + 1) / 2 - 1)
        self._pos[1] = int((self._Env.size + 1) / 2 - 1)
    '''
        get the next state based on a state and an action
        s is a np.array type representing the state
    
    '''
    def _move(self,action,s):
        res = s+Actions[str(action)]
        if res[0] < 0 or res[0] > self._Env.size-1 or res[1] <0 or res[1]>self._Env.size-1:
            return s
        else:
            return res

    '''
    Is a terminal state
    '''
    def _is_terminal(self,s):
        if np.array_equal(s,np.array([self._Env.size-1,self._Env.size-1])) or np.array_equal(s,np.array([0,0])):
            return True
        else:
            return False
    '''
    The function is to return the pairs of probable next state and its probability given the current state
    and the current action 
    the action otained here is the post-rolling one
    the return is a dict {"action": probability }
    action here is an int type
    '''
    def _get_direction(self,action, s):
        res = dict()
        print(Actions)
        for k,v in Actions.items():
            if int(k) != action:
                res[str(k)]=self._Env.distribution[1]/4
            else:
                res[str(action)]=self._Env.distribution[0] + self._Env.distribution[1]/4
        return res
    '''
    Given i,j, a state here, to calculate its state value
    '''
    def _run_bellman(self,s):
        res = 0
        for action in range(4):
            # given the action and i,j
            next_direction = self._get_direction(action,s)
            for key,value in next_direction.items():
                next_state=self._move(int(key),s)
                res += Action_prob*value*(self._Env.reward + Gamma*self._state_value[next_state[0]][next_state[1]])
        return res

    '''
    Policy evaluation 
    '''
    def policy_evaluation(self):
        theta=0.001
        while True:
            delta = 0
            for i in range(self._Env.size):
                for j in range(self._Env.size):
                    current_state=np.array([i,j])
                    if self._is_terminal(current_state):
                        continue
                    self._state_value[i][j] = self._run_bellman(current_state)
            if delta<theta:
                break
def main():
    Environment = Grid(5,[0.9,0.1])
    A = Agent(Environment)
    A.policy_evaluation()

main()