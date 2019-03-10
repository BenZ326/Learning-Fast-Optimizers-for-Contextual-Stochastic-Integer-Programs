
# codes for assignment 2
# Authors: Xiangyi Zhang and Rahul Patel
# Date: 2019/3/8

import matplotlib
import numpy as np
import random as rd
import gym
import matplotlib.pyplot as plt
import Tiling
import copy
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
        self.v = state[2]
        self.state = Tiling.tile_encode(state,tilings, flatten=False)
        self.tilings = tilings
        self.lamb = lamb
        self.gamma = gamma
        self.alpha = alpha


    """"
    Generate the policy to be evaluated
    torque in the same direction as the current velocity with probability 0.9
    torque the opposite direction as the current velocity with probability 0.1
    torque whichever direction if the current velocity is 0
    """
    def _fix_policy(self):
        action = env.action_space.sample()
        prob = rd.random()
        if self.v>0:
            action = abs(action) if prob>=0.1 else -1*abs(action)
            return action
        if self.v<0:
            action = -1*abs(action) if prob >=0.1 else abs(action)
            return action
        return action

    """"
    based on the current state and a returned action, it goes to the next state
    """
    def move(self):
        action = self._fix_policy()
        s, r, done, info = self.env.step(action)
        self.state = Tiling.tile_encode(s,tilings, flatten=False)
        self.v = s[2]

    """
    Monte carlo method 
    """
    def MC(self):
        MC_Vtable = Agent.TiledV_table(self.tilings)
        episode_n = 200
        time_step = 0
        for i_episode in range(1,episode_n+1):
            episode = []
            self.env.reset()
            episode.append((self.state, time_step, 0))
            while(True):
                action = self._fix_policy()
                state, reward, done, info = self.env.step(action)
                self.v = state[2]                                 #update the velocity
                self.state = Tiling.tile_encode(state,self.tilings, flatten=False)
                time_step += 1
                episode.append((self.state,time_step, reward))
                if done:
                    break
            for idx_tiling in range(len(MC_Vtable.v_table)):
                self.tiling_wise_update_MC(idx_tiling, episode, MC_Vtable)

            """"
            Average the values over all tilings
            """
        MC_Vtable.average_tilings()
        MC_Vtable.plot_value_table(0,"MC value")

    """
    TD learning & Eligibility Trace & Linear Function Approximation
    """
    def TD_ET(self):
        TD_Vtable = Agent.TiledV_table(self.tilings)
        TD_Etable = Agent.Eligibility(self.tilings,self.lamb, self.alpha, self.gamma)
        episode_n = 200
        for episode in range(1,episode_n+1):
            time_step = 0
            self.v = 0
            self.env.reset()
            while(True):
                action = self._fix_policy()
                old_state = copy.deepcopy(self.state)
                new_state, reward, done, _ = self.env.step(action)
                self.v = new_state[2]           # record the velocity before being tile encoded
                self.state = Tiling.tile_encode(new_state,self.tilings,flatten=False)
                for idx_tiling in range(len(TD_Vtable.v_table)):
                    TD_Etable.tiling_wise_update_E(idx_tiling,self.state[idx_tiling])
                    self.tiling_wise_update_TD(idx_tiling,old_state[idx_tiling],TD_Vtable,reward,TD_Etable)
                if done:
                    break
                time_step += 1

        TD_Vtable.average_tilings()
        TD_Vtable.plot_value_table(0,"TD Value")

    """"
    TD update state value 
    """
    def tiling_wise_update_TD(self,idx, old_state, TD_vtable, reward,TD_etable):
        td_error = reward + self.gamma*TD_vtable.v_table[idx][self.state[idx]]-TD_vtable.v_table[idx][old_state]
        TD_vtable.v_table[idx] = np.add(TD_vtable.v_table[idx],self.alpha*td_error*TD_etable.E_table[idx])
    """"
    tiling wise update a state value table
    """
    def tiling_wise_update_MC(self, idx, episode,MC_vtable):
        state_in_episode = set([tuple(x[0][idx]) for x in episode])
        for s in state_in_episode:
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                           if x[0][idx] == s)
            G = sum([x[2]*(self.gamma**i) for i,x in enumerate(episode[first_occurence_idx+1:])])
            tmp_sum = MC_vtable.v_table[idx][s]*MC_vtable.counter_table[idx][s] + G
            MC_vtable.counter_table[idx][s] += 1
            MC_vtable.v_table[idx][s] = tmp_sum/MC_vtable.counter_table[idx][s]



    """"
    Inner class: state_value table
    """
    class TiledV_table:
        def __init__(self, tilings):
            self.tilings = tilings
            state_sizes = [tuple(len(split)+1 for split in tile) for tile in self.tilings]
            self.v_table = [self._sub_table(state_size)  for state_size in state_sizes]
            self.v_output_table = self._sub_table(state_sizes[0])
            self.counter_table = [self._sub_table(state_size)  for state_size in state_sizes]


        def _sub_table(self, state_size):
            return np.zeros(state_size)

        def average_tilings(self):
            print(self.v_output_table.shape)
            x,y,z = self.v_output_table.shape
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        self.v_output_table[(i,j,k)] = np.mean([T[(i,j,k)] for T in self.v_table])

        def plot_value_table(self,velocity,title):
            """
            Plots the value table as a surface plot.
            """
            x,y,_ = self.v_output_table.shape
            min_x = 0
            max_x = x
            min_y = 0
            max_y = y

            x_range = np.arange(min_x, max_x,step=1)
            y_range = np.arange(min_y, max_y,step=1)
            X, Y = np.meshgrid(x_range, y_range)

            # Find value for all (x, y) coordinates
            Value = self.v_output_table[:,:,velocity]
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

            plot_surface(X, Y, Value, "{}, velocity = {}".format(title,velocity))

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
            self.E_table[tiling_idx] *= self.lamb*self.gamma
            self.E_table[tiling_idx][state] += 1


    """"
    Weights for linear function approximation
    """


env = gym.make('Pendulum-v0')
env.seed(505)

observation = env.observation_space
low = observation.low
high = observation.high
action_low=env.action_space.low
action_high=env.action_space.high
n = 10
tiling_specs = [((n, n,n), (-0.2, -0.15, -0.2)),
                ((n, n,n), (0.0, 0.0, 0.0)),
                ((n, n,n), (0.2, 0.15, 0.2))]
tilings = Tiling.create_tilings(low, high, tiling_specs)

initial_state = np.zeros(3)

lamb = 1.0
gamma =0.8
alpha = 1/200
Learning = Agent(initial_state,env,tilings,lamb, gamma, alpha)
Learning.MC()
Learning.TD_ET()

