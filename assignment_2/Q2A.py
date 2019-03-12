
# codes for assignment 2
# Authors: Xiangyi Zhang and Rahul Patel
# Date: 2019/3/8

import matplotlib
import numpy as np
import random as rd
import gym
import matplotlib.pyplot as plt
import Tiling
import math
from mpl_toolkits.mplot3d import Axes3D




"""
tilings_specs_actions=[([n],[-0.2]),
                      ([n],[0.0]),
                      ([n],[0.2])]

action_tilings = Tiling.create_tilings(action_low, action_high, tilings_specs_actions)
"""

class Agent:

    """
    the input state here is not discretilized
    v is the velocity of the current pendulum
    """
    def __init__(self, state, env,tilings, lamb, gamma, alpha):
        self.env =env
        self.v = state[1]
        self.state = Tiling.tile_encode(state,tilings, flatten=False)
        self.tilings = tilings
        self.lamb = lamb
        self.gamma = gamma
        self.alpha = alpha

    """"
    Here we use the angular position and angular velocity as our states, so we need to 
    transfer the cos, sin to angular position
    
    if sin/cos > 
    """


    """"
    Generate the policy to be evaluated
    torque in the same direction as the current velocity with probability 0.9
    torque the opposite direction as the current velocity with probability 0.1
    torque whichever direction if the current velocity is 0
    """
    def _fix_policy(self):
        #action = env.action_space.sample()
        action = np.array([env.unwrapped.max_torque])
        prob = np.random.rand(1)
        if self.v>0:
            action = abs(action) if prob>=0.1 else -1*abs(action)
            return action
        if self.v<0:
            action = -1*abs(action) if prob >=0.1 else abs(action)
            return action
        if self.v == 0:
            action = -1*abs(action) if prob >=0.5 else abs(action)
            return action

    """
    TD learning & Eligibility Trace & Linear Function Approximation
    """
    def TD_ET(self):

        episode_n = 200
        TD_Etable = Agent.Eligibility(self.tilings, self.lamb, self.alpha, self.gamma)
        TD_Wtable = Agent.Weights(self.tilings)
        estimated_value=np.zeros(episode_n)
        for episode in range(0,episode_n):
            TD_Etable.reset()
            time_step = 0
            env.reset()
            initial_state = get_state(reset_env(self.env))
            self.v = initial_state[1]
            self.state = Tiling.tile_encode(initial_state, tilings, flatten=False)
            while(True):
                action = self._fix_policy()
                old_state_estivalue = TD_Wtable.Estimated_Value(self.state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = get_state(new_state)
                #update eligibility trace
                for idx_tiling in range(len(TD_Etable.E_table)):
                    TD_Etable.tiling_wise_update_E(idx_tiling, self.state[idx_tiling])
                #update states
                self.v = new_state[1]           # record the velocity before being tile encoded
                self.state = Tiling.tile_encode(new_state,self.tilings,flatten=False)
                new_state_estivalue = TD_Wtable.Estimated_Value(self.state)
                td_error = reward + self.gamma * new_state_estivalue - old_state_estivalue
                print("td error = ", td_error)
                # update Weights
                for idx_tiling in range(len(TD_Wtable.W_table)):
                    TD_Wtable.W_table[idx_tiling] += self.alpha * td_error * (TD_Etable.E_table[idx_tiling])
                if done:
                    break
                time_step += 1
            estimated_value[episode] = TD_Wtable.Estimated_Value(Tiling.tile_encode(np.zeros(2),self.tilings,flatten=False))
        plt.plot(range(episode_n),estimated_value)
        plt.show()
        #TD_Wtable.plot_value_function()
        #TD_Vtable.plot_value_table(0,"TD Value")



    """"
    Eligibility 
    """
    class Eligibility:
        def __init__(self,tilings,lamb,alpha,gamma):
            """"
            For each tiling, we maintain the eligibility for each state
            """
            self.lamb = lamb
            self.alpha = alpha
            self.gamma = gamma
            state_sizes = [tuple(len(split)+1 for split in tile)  for tile in tilings]
            self.E_table = [ self._create_sub_table(state_size) for state_size in state_sizes]


        def _create_sub_table(self,state_size):
            return np.zeros(state_size)

        """"
        Based on the current state, update the eligibility, tiling wise update 
        """
        def tiling_wise_update_E(self,tiling_idx,state):
            self.E_table[tiling_idx][state] *= self.lamb*self.gamma
            self.E_table[tiling_idx][state] += 1

        def reset(self):
            for idx in range(len(self.E_table)):
                self.E_table[idx] *=0


    """"
    Weights for linear function approximation
    The feature vector is equivalent to the state, since we have three parameters to describe a state,
    so for each tiling, for each state, we need three weights.
    """
    class Weights:
        """"
        """
        def __init__(self,tilings):
            self.tilings = tilings
            state_sizes = [tuple(len(split)+1 for split in tile) for tile in tilings]
            self.W_table = [self._create_sub_table(state_size) for state_size in state_sizes]
        #
        """"
        The second argument is the number of parameters we use to describe a state
        the function generates a weights vector for each tiling
        """
        def _create_sub_table(self,state_size):
            # sample from U[-0.001,0.001]
            np.random.seed(100)
            return (np.random.random_sample(state_size)-0.5)*(1/500)
            #return np.zeros(state_size)
        """
        The function is to update the weights in an incremental fassion.
        Require:
        1. tiling idx to specify a certain tiling
        2. td_error is the TD error at the current time step 
        3. alpha: learning rate
        4. TD_Etable: eligibility table
        """
        """"
        @states: encoded states
        return the estimated value based on the states
        """
        def Estimated_Value(self,states):
            result = np.sum(self.W_table[idx][x] for idx,x in enumerate(states))
            return result

        def plot_value_function(self):
            x_min = -1*np.pi
            x_max = np.pi
            y_min = -8
            y_max = 8

            x_range = np.arange(x_min, x_max,step=0.1)
            y_range = np.arange(y_min, y_max,step=0.1)
            X, Y = np.meshgrid(x_range, y_range)
            Value = np.zeros(X.shape)
            x_size, y_size = X.shape
            for i in range(x_size):
                for j in range(y_size):
                    tmp_state = np.array([X[i][j],Y[i][j]])
                    encoded_state = Tiling.tile_encode(tmp_state,self.tilings)
                    Value[i][j] = self.Estimated_Value(encoded_state)

            def plot_surface(X, Y, Z, title):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                       cmap=matplotlib.cm.coolwarm,linewidth=0,antialiased=False)
                ax.set_xlabel('State--X')
                ax.set_ylabel('State--Y')
                ax.set_zlabel('Value')
                ax.set_title(title)
                ax.view_init(ax.elev, -120)
                fig.colorbar(surf)
                plt.show()

            plot_surface(X,Y,Value,"TD_Function_Approximation")


""""
transfer the env default state to angular position, angular speed
"""

def get_state(_env_state):
    cosx= _env_state[0]
    sinx= _env_state[1]
    velocity= _env_state[2]
    theta = math.acos(cosx) if sinx >=0 else -1*math.acos(cosx)
    return np.array([theta, velocity])

def reset_env(env):
    env.unwrapped.state = np.array([0,0])
    env.unwrapped.last_u = None
    return env.unwrapped._get_obs()


env = gym.make('Pendulum-v0')
env.seed(200)


observation = env.observation_space
low = np.array([-1*np.pi, -8])
high = np.array([1*np.pi,8])
action_low=env.action_space.low
action_high=env.action_space.high
n = 10
tiling_specs = [((n, n), (-0.2, -0.2)),
                ((n,n),(-0.1,-0.1)),
                ((n, n), (0.0, 0.0)),
                ((n, n), (0.1, 0.1)),
                ((n,n),(0.2, 0.2))]
tilings = Tiling.create_tilings(low, high, tiling_specs)



lamb = 1.0
gamma =0.95
alpha = 1/20
intial_obsv  = reset_env(env)

Learning = Agent(intial_obsv ,env,tilings,lamb, gamma, alpha)
Learning.TD_ET()

