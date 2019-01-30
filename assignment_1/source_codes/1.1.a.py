# Necessary imports
import heapq
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from operator import add


class UCBBandit:
    """
    UCBBandit
    """
    def __init__(self, arms, delta, eps,total_steps):
        self.arms = arms
        self.eps = eps
        self.delta = delta
        self.est_dstr = []
        self.reward_history=np.zeros(total_steps)
        self.optimal_ratio=np.zeros(total_steps)
        self.total_steps=total_steps
        for i in range(arms):
            self.est_dstr.append([])



    def _calculate_c(self):
        """
        Calculate the C_(i,t) = 2U(T_i(t), delta/n)
        """
        res = []
        for arm, _ in enumerate(self.q_est):
            exp_1 = 1 + self.eps ** 0.5
            exp_2 = 1 + self.eps
            exp_3 = (1 + self.eps) * self.N[arm]
            exp = exp_1 * \
                  np.sqrt((exp_2 * np.log((np.log(exp_3 + 2)) / self.delta)) / (self.N[arm]))
            res.append(exp)
        return res

    def initialize(self,q_true):
        """
        Initializes the UCBBandit with default values before each trial
        """
        self.q_true = q_true
        self.q_est = np.zeros(self.arms)
        self.N = np.zeros(self.arms)
        self.history = []
        self.time = 0
        self.best_arm=np.argmax(self.q_true)
        self.optimal_current=np.zeros(self.total_steps)
        self.reward_history_current=np.zeros(self.total_steps)
        self.H1 = 0
        for idx in range(1,self.arms):
            self.H1 = (1/(self.q_true[0]-self.q_true[idx])**2)+ self.H1
        for arm, _ in enumerate(self.q_est):
            self.play_arm(arm)

    def select_arm(self):
        """
        Select the arm to play for the Bandit
        """
        h = np.argmax(self.q_est)
        C = self._calculate_c()
        QC = np.sum([self.q_est, C], axis=0)
        l_ = heapq.nlargest(2, range(self.arms), key=QC.__getitem__)
        l = l_[0] if h != l_[0] else l_[1]
        # Select arm
        arm = l_[0]
        # Check stopping criterion
        if self.q_est[h] - C[h] > self.q_est[l] + C[l]:
            self.extend_history(h)
            return arm, True
        return arm, False

    def play_arm(self, arm):
        """
        Play the arm and update its value estimate
        """
        self.time += 1
        self.N[arm] += 1.0
        reward = np.random.normal(self.q_true[arm], 0.5)
        self._update_opt_ratio(arm)
        self._update_av_reward(reward)
        self.sample_history(arm)
        self.q_est[arm] += 1.0 / self.N[arm] * (reward - self.q_est[arm])

    def sample_history(self, arm):
        """
        Record the history of sampling
        """
        self.history.append(arm)

    def extend_history(self,output_arm):
        """
        When the algorithm terminates before it hits total time steps
        """
        while self.time<self.total_steps:
            self.time += 1
            self.history.append(output_arm)
            reward = np.random.normal(self.q_true[output_arm], 0.5)
            self._update_opt_ratio(output_arm)
            self._update_av_reward(reward)

    def get_estimate_probability(self):
        """
        get accumulated Pr{I_t=i} up till now
        """
        for t in range(0,self.total_steps):
            left_index = max(0,t-self.arms+1)
            right_index = t
            for arm in range(self.arms):
                prob=self.history[left_index:right_index+1].count(arm)/(right_index-left_index+1)
                if t >= len(self.est_dstr[arm]):
                    self.est_dstr[arm].append(prob)
                else:
                    self.est_dstr[arm][t] += prob

    def average_est_dstr(self,trials):
        """
         average the est_dstr
         """
        for arm in range(self.arms):
            self.est_dstr[arm]=list(map((1/trials).__mul__, self.est_dstr[arm]))

    def learning(self):
        for step in range(self.total_steps):
            arm,done = self.select_arm()
            if not done:
                self.play_arm(arm)
            else:
               #print("The best arm is {}".format(arm))
               break
        self.optimal_ratio= self.optimal_ratio+self.optimal_current
        self.reward_history=self.reward_history+self.reward_history_current
        self.get_estimate_probability()
        return



    def average_reward_history(self,trials):
        """
         average reward_history
         """
        res=np.zeros(self.total_steps)
        for t in range(0,self.total_steps):
            left_index = max(0,t-self.arms+1)
            right_index = t
            avg=sum(self.reward_history[left_index:right_index+1])/(right_index-left_index+1)
            res[t] += avg
        self.reward_history = res / trials



    def average_optimal_ratio(self,trials):
        """
         average reward_history
         """
        res = np.zeros(self.total_steps)
        for t in range(0,self.total_steps):
            left_index = max(0,t-self.arms+1)
            right_index = t
            avg=sum(self.optimal_ratio[left_index:right_index+1])/(right_index-left_index+1)
            res[t] += avg
        self.optimal_ratio = res / trials

    def _update_opt_ratio(self,arm):
        if arm==self.best_arm and self.time<=self.total_steps:
            self.optimal_current[self.time-1] += 1
    def _update_av_reward(self,reward):
        if self.time<=self.total_steps:
            self.reward_history_current[self.time-1] += reward

    def plot1(self):
        line0, = plt.plot(range(self.total_steps) / self.H1, self.est_dstr[0], '-', color='b', markersize=2,
                          label="$\mu_0=1.0$")
        line1, = plt.plot(range(self.total_steps) / self.H1 , self.est_dstr[1], '-', color='g', markersize=2,
                          label="$\mu_1=0.8$")
        line2, = plt.plot(range(self.total_steps) / self.H1 , self.est_dstr[2], '-', color='r', markersize=2,
                          label="$\mu_2=0.6$")
        line3, = plt.plot(range(self.total_steps) / self.H1 , self.est_dstr[3], '-', color='c', markersize=2,
                          label="$\mu_3=0.4$")
        line4, = plt.plot(range(self.total_steps) / self.H1, self.est_dstr[4], '-', color='y', markersize=2,
                          label="$\mu_4=0.2$")
        line5, = plt.plot(range(self.total_steps) / self.H1, self.est_dstr[5], '-', color='k', markersize=2,
                          label="$\mu_5=0.0$")
        plt.xlabel("Number of pulls (units of H1)")
        plt.ylabel("$P(I_t=i)$")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend(bbox_to_anchor=(0, 0.5, 0.4, 0.5))
        plt.show()
    def plot2(self,color,alg_name):
        line0, = plt.plot(self.reward_history, '-', color=color, markersize=2,
                          label=alg_name)
        plt.xlabel("Number of pulls (Absolute)")
        plt.ylabel("Average Reward")
        plt.yticks(np.arange(0, 1.5, 0.2))
        plt.legend(bbox_to_anchor=(0, 0.5, 0.4, 0.5))
    def plot3(self,color,alg_name):
        line0, = plt.plot( self.optimal_ratio, '-', color=color, markersize=2,
                          label=alg_name)
        plt.xlabel("Number of pulls (Absolute)")
        plt.ylabel("Optimal Action %")
        plt.yticks(np.arange(0, 1.0, 0.1))
        plt.legend(bbox_to_anchor=(0, 0.5, 0.4, 0.5))


class AEBandit(UCBBandit):
    def __init__(self, arms, delta, eps,total_steps,r_k):
        UCBBandit.__init__(self,arms,delta,eps,total_steps)
        self.r_k = r_k

    def initialize(self,q_true):
        """
        Initializes the UCBBandit with default values before each trial
        """
        self.q_true=q_true
        self.q_est = np.zeros(self.arms)
        self.N = np.zeros(self.arms)
        self.history = []
        self.time = 0
        self.optimal_current = np.zeros(self.total_steps)
        self.reward_history_current = np.zeros(self.total_steps)
        self.Omega = set()
        self.best_arm=np.argmax(q_true)
        self.H1 = 0
        for arm in range(self.arms):
            self.Omega.add(str(arm))

    def reference_arm(self):
        return np.argmax(self.q_est)

    def _calculate_c(self,arm):
        exp_1 = 1 + self.eps ** 0.5
        exp_2 = 1+self.eps
        exp_3 = (1 + self.eps) * self.N[arm]
        exp = exp_1 * \
              np.sqrt((exp_2 * np.log((np.log(exp_3+2)) / (self.delta ))) / (self.N[arm]))
        return exp

    def update_omega(self,reference_arm):
        removed_arms=set()
        for arm in self.Omega:
            C_a=self._calculate_c(reference_arm)
            C_i=self._calculate_c(int(arm))
            if self.q_est[reference_arm]-C_a >= self.q_est[int(arm)]+C_i:
                removed_arms.add(arm)
        self.Omega = self.Omega-removed_arms

    def learning(self):
        while self.time < self.total_steps:
            #print(self.time, self.total_steps)
            for arm in self.Omega:
                self.play_arm(int(arm))
            #if self.time==1499:
             #   print(self.time)
            reference_arm = self.reference_arm()
            self.update_omega(reference_arm)
        self.optimal_ratio= self.optimal_ratio+self.optimal_current
        self.reward_history=self.reward_history+self.reward_history_current
        self.get_estimate_probability()
        return



class LUCBBandit(UCBBandit):
    def select_arm(self):
        """
        Select the arm to play for the Bandit
        """
        h = np.argmax(self.q_est)
        C = self._calculate_c()
        QC = np.sum([self.q_est, C], axis=0)
        l_ = heapq.nlargest(2, range(self.arms), key=QC.__getitem__)
        l = l_[0] if h != l_[0] else l_[1]
        # Select arm
        # Check stopping criterion
        if self.q_est[h] - C[h] > self.q_est[l] + C[l]:
            self.extend_history(h)
            return h,l, True
        return h,l,False

    def learning(self):
        while self.time <= self.total_steps:
            arm1, arm2, done = self.select_arm()
            if not done:
                self.play_arm(arm1)
                self.play_arm(arm2)
            else:
                #print("The best arm is {}".format(arm1))
                break
        self.optimal_ratio= self.optimal_ratio+self.optimal_current
        self.reward_history=self.reward_history+self.reward_history_current
        self.get_estimate_probability()



def main():
    """
    Main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--r_k', type=int, default=1)
    args = parser.parse_args()
    bandit_LUCB = LUCBBandit(arms=10, delta=args.delta, eps=args.epsilon, total_steps=args.steps)
    bandit_UCB = UCBBandit(arms=10, delta=args.delta, eps=args.epsilon, total_steps=args.steps)
    bandit_AE = AEBandit(arms=10, delta=args.delta, eps=args.epsilon,total_steps=args.steps,r_k=args.r_k)
    for trial in range(args.trials):
        q_true = np.random.normal(0,0.12,10)
        bandit_AE.initialize(q_true)
        bandit_AE.learning()
        bandit_UCB.initialize(q_true)
        bandit_UCB.learning()
        bandit_LUCB.initialize(q_true)
        bandit_LUCB.learning()
    bandit_AE.average_est_dstr(args.trials)
    bandit_AE.average_reward_history(args.trials)
    bandit_AE.average_optimal_ratio(args.trials)
    bandit_UCB.average_est_dstr(args.trials)
    bandit_UCB.average_reward_history(args.trials)
    bandit_UCB.average_optimal_ratio(args.trials)
    bandit_LUCB.average_est_dstr(args.trials)
    bandit_LUCB.average_reward_history(args.trials)
    bandit_LUCB.average_optimal_ratio(args.trials)
    return bandit_AE, bandit_UCB, bandit_LUCB
AE,UCB,LUCB=main()

AE.plot2('b',"AE")
UCB.plot2('r',"UCB")
LUCB.plot2('g',"LUCB")
plt.show()
AE.plot3('b',"AE")
UCB.plot3('r',"UCB")
LUCB.plot3('g',"LUCB")
plt.show()





