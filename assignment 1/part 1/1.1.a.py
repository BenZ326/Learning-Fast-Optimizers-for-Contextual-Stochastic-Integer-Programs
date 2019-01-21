# Necessary imports
import heapq
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class UCBBandit:
    """
    UCBBandit
    """

    def __init__(self, arms, delta, eps):
        self.q_true = list(reversed(np.linspace(0, 1, arms)))
        self.arms = arms
        self.eps = eps
        self.delta = delta
        self.time = 0

    def _calculate_c(self):
        """
        Calculate the C_(i,t) = U(t, delta/2)
        """
        exp_1 = 1 + self.eps**0.5
        exp_2 = (1+self.eps)*self.time
        exp = exp_1 * \
            np.sqrt((exp_2 * np.log(np.log(exp_2) / self.delta)) / (2*self.time))
        return exp

    def initialize(self):
        """
        Initializes the UCBBandit with default values before each trial
        """
        self.q_est = np.zeros(self.arms)
        self.N = np.zeros(self.arms)
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
            return arm, True
        return arm, False

    def play_arm(self, arm):
        """
        Play the arm and update its value estimate
        """
        self.time += 1.0
        self.N[arm] += 1.0

        reward = np.random.normal(self.q_true[arm], 0.5)
        self.q_est[arm] += 1.0 / self.N[arm] * (reward - self.q_est[arm])


def main():
    """
    Main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--trials', type=int, default=5000)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    bandit = UCBBandit(arms=6, delta=args.delta, eps=args.epsilon)
    for trial in range(args.trials):
        bandit.initialize()
        for step in range(args.steps):
            arm, done = bandit.select_arm()
            if not done:
                bandit.play_arm(arm)
            else:
                print("The best arm is {}".format(arm))
                break


if __name__ == '__main__':
    main()
