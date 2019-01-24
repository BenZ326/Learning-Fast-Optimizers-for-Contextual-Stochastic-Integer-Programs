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
        self.q_true = list(reversed(np.linspace(0, 1, arms)))
        self.arms = arms
        self.eps = eps
        self.delta = delta
        self.est_dstr = []
        self.total_steps=total_steps
        for i in range(arms):
            self.est_dstr.append([])



    def _calculate_c(self):
        """
        Calculate the C_(i,t) = 2U(T_i(t), delta/n)
        """
        res = []
        for arm, _ in enumerate(self.q_est):
            #exp_1 = 1 + self.eps ** 0.5
            #exp_2 = (1 + self.eps) * self.N[arm]
            #exp = exp_1 * \
            #     np.sqrt((exp_2 * np.log((np.log(exp_2)+2) / self.delta )) / (2 * self.N[arm]))
            exp=1.05*0.25/np.log(self.N[arm])
            res.append(exp)
        return res

    def initialize(self):
        """
        Initializes the UCBBandit with default values before each trial
        """
        self.q_est = np.zeros(self.arms)
        self.N = np.zeros(self.arms)
        self.history = []
        self.time = 0
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
        self.time += 1.0
        self.N[arm] += 1.0

        reward = np.random.normal(self.q_true[arm], 0.5)
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
        while len(self.history)<self.total_steps:
            self.history.append(output_arm)

    def get_estimate_probability(self):
        """
        get accumulated Pr{I_t=i} over trials
        """
        for t in range(0,self.total_steps):
            left_index=max(0,t-self.arms+1)
            right_index=t
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




class AEBandit(UCBBandit):
    def __init__(self, arms, delta, eps,total_steps,r_k):
        self.q_true = list(reversed(np.linspace(0, 1, arms)))
        self.arms = arms
        self.eps = eps
        self.delta = delta
        self.est_dstr = []
        self.total_steps=total_steps
        for i in range(arms):
            self.est_dstr.append([])
        self.r_k = r_k

    def initialize(self):
        """
        Initializes the UCBBandit with default values before each trial
        """
        self.q_est = np.zeros(self.arms)
        self.N = np.zeros(self.arms)
        self.history = []
        self.time = 0
        self.Omega = set()
        for arm in range(self.arms):
            self.Omega.add(str(arm))

    def reference_arm(self):
        return np.argmax(self.q_est)

    def _calculate_c(self,arm):
        exp_1 = 1 + self.eps ** 0.5
        exp_2 = 1+self.eps
        exp_3 = (1 + self.eps) * self.N[arm]
        exp = exp_1 * \
        np.sqrt((exp_2 * np.log((np.log(exp_3+1)) / self.delta )) / ( self.N[arm]))
        return 2*exp

    def update_omega(self,reference_arm):
        removed_arms=set()
        for arm in self.Omega:
            C_a=self._calculate_c(reference_arm)
            C_i=self._calculate_c(int(arm))
            if self.q_est[reference_arm]-C_a >= self.q_est[int(arm)]+C_i:
                removed_arms.add(arm)
        self.Omega = self.Omega-removed_arms

    def learning(self):
        #while len(self.Omega)>1:
        for i in range(self.total_steps):
            for arm in self.Omega:
                self.play_arm(int(arm))
            reference_arm = self.reference_arm()
            self.update_omega(reference_arm)
        return





def main():
    """
    Main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--trials', type=int, default=5000)
    parser.add_argument('--steps', type=int, default=7000)
    parser.add_argument('--r_k', type=int, default=1)
    args = parser.parse_args()

    #bandit = UCBBandit(arms=6, delta=args.delta, eps=args.epsilon,total_steps=args.steps)
    bandit_AE = AEBandit(arms=6, delta=args.delta, eps=args.epsilon,total_steps=args.steps,r_k=args.r_k)
    for trial in range(args.trials):
        #bandit.initialize()
        bandit_AE.initialize()
        bandit_AE.learning()
        #for step in range(args.steps):
            #arm, done = bandit.select_arm()
            #if not done:
            #    bandit.play_arm(arm)
            #else:
            #   print("The best arm is {}".format(arm))
            #    break
        #bandit.get_estimate_probability()
        bandit_AE.get_estimate_probability()
    bandit_AE.average_est_dstr(args.trials)
    return bandit_AE


b=main()
plt.plot(range(len(b.est_dstr[0])),b.est_dstr[0],'.',color='b')
plt.plot(range(len(b.est_dstr[1])),b.est_dstr[1],'.', color='g')
plt.plot(range(len(b.est_dstr[2])),b.est_dstr[2],'.',color='r')
plt.plot(range(len(b.est_dstr[3])),b.est_dstr[3],'.',color='c')
plt.plot(range(len(b.est_dstr[4])),b.est_dstr[4],'.',color='y')
plt.plot(range(len(b.est_dstr[5])),b.est_dstr[5],'.',color='k')
plt.show()
print(b.history)
#if __name__ == '__main__':
#    main()
