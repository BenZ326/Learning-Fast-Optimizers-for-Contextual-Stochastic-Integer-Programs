
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


import Function_Approximator





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
    def __init__(self,env, lamb, gamma, alpha, function_approximator,episode_n):
        self.env =env
        self.lamb = lamb
        self.gamma = gamma
        self.alpha = alpha
        self.state = np.zeros(2)
        self.approximator = function_approximator
        self.episode_n = episode_n

    def _reset_state(self):
        self.state = np.zeros(2)
        self.env.state =np.zeros(2)
    """"
    Here we use the angular position and angular velocity as our states, so we need to 
    transfer the cos, sin to angular position
>>>>>>> xyz-dev
    """


    """"
    Generate the policy to be evaluated
    torque in the same direction as the current velocity with probability 0.9
    torque the opposite direction as the current velocity with probability 0.1
    torque whichever direction if the current velocity is 0
    """
    """"
    Vanilla MC
    """

    def MC(self):
        env.reset()
        max_time_step = 200
        values = np.zeros(self.episode_n)
        for episode in range(self.episode_n):
            time_step =0
            value = 0
            self._reset_state()
            while(time_step < max_time_step):
                action = self._fix_policy()
                new_state, reward, done, _ =self.env.step(action)
                value += reward*(self.gamma**time_step)
                self.state = get_state(new_state)
                time_step += 1
            values[episode] = value
        plt.plot(range(self.episode_n),values)
        plt.title("Monte Carlo Method")
        plt.xlabel("# episode")
        plt.ylabel("estimated value")
        plt.show()


    def _fix_policy(self):
        #action = env.action_space.sample()
        action = np.array([env.unwrapped.max_torque])
        prob = np.random.rand(1)
        self.v = self.state[1]

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
    run an episode
    """
    def TD_ET(self):
            env.reset()
            eligibility_trace = np.zeros_like(self.approximator.weights)
            self._reset_state()
            max_time_step = 400
            time_step = 0
            while( time_step < max_time_step):
                action = self._fix_policy()
                state_prime, reward, done, _ = self.env.step(action)
                state_prime = get_state(state_prime)
                eligibility_trace *= self.gamma*self.lamb
                eligibility_trace += self.approximator.get_gradient(self.state)
                td_error = reward + self.gamma * self.approximator.estimated_value(state_prime) - self.approximator.estimated_value(self.state)
                #print("td error = ", td_error)
                delta_weights = self.alpha*td_error*eligibility_trace
                self.approximator.update_weights(delta_weights)
                self.state = state_prime
                # update Weights
                if done:
                    break
                time_step += 1



    def Experiments(self,target_state):
        run_time = 10
        state_values = np.zeros([run_time,self.episode_n+1])
        for run in range(run_time):
            self.approximator.reset_weights()
            state_values[run][0] = self.approximator.estimated_value(target_state)
            for episode in range(1,self.episode_n+1):
                self.TD_ET()
                state_values[run][episode] = self.approximator.estimated_value(target_state)
        return np.mean(state_values,axis=0)
        #plt.plot(range(0,episode_n+1),np.mean(state_values,axis =0))
        #plt.show()



""""
transfer the env default state to angular position, angular speed
"""

def get_state(_env_state):
    cosx= _env_state[0]
    sinx= _env_state[1]
    velocity= _env_state[2]
    theta = math.acos(cosx) if sinx >=0 else -1*math.acos(cosx)
    return np.array([theta, velocity])

seed = 500
env = gym.make('Pendulum-v0').env
env.seed(seed)
>>>>>>> xyz-dev


observation = env.observation_space
low = np.array([-1*np.pi, -8])
high = np.array([1*np.pi,8])
action_low=env.action_space.low
action_high=env.action_space.high
n = 10
tiling_specs = [((n, n), (-0.1, +0.33)),
                ((n,n),(-0.05,+0.15)),
                ((n, n), (0.0, 0.0)),
                ((n, n), (0.05, -0.15)),
                ((n,n),(0.1, -0.33))]

tilings = Tiling.create_tilings(low, high, tiling_specs)


gamma =0.90

#lamb_set = [0,0.3,0.7,0.9,1.0]
#alpha_set = [1/20,1/40,1/80]
lamb_set = [0.9]
alpha_set =[1/20]


episode_n = 200

for lamb in lamb_set:
    result = np.zeros([len(alpha_set), episode_n+1])
    for idx,alpha in enumerate(alpha_set):
        Linear_Func = Function_Approximator.linear_function_approximator(tilings, -0.001,0.001,seed)
        Learning = Agent(env,lamb,
                 gamma, alpha, Linear_Func, episode_n
                 )
        result [idx] = Learning.Experiments(np.zeros(2))
        plt.plot(range(0,episode_n+1),result[idx],label=r"$\alpha$"+f"={alpha}")
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Estimated Value")
        plt.title(f"Estimated value of state {(0,0)} when $\lambda$={lamb}")
    plt.show()

""""
Linear_Func = Function_Approximator.linear_function_approximator(tilings, -0.001, 0.001, seed)
Learning = Agent(env, lamb_set[0],
                 gamma, alpha_set[0], Linear_Func, episode_n
                 )
Learning.MC()
"""





