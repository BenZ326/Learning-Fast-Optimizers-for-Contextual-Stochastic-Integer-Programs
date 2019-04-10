from environment import Env_KS
from instance import instance_generator
from initialisation_policy import InitializationPolicy
from initialisation_policy import Baseline

import torch as T
import argparse
import numpy as np

##################################################################
#                       DEFAULT PARAMETERS                       #
##################################################################
DEFAULT = {}
DEFAULT["EPOCHS"] = 1000
DEFAULT["PROBLEM"] = "ks"
DEFAULT["IS_PENALTY_SAME"] = True
DEFAULT["DIM_HIDDEN"] = 10
DEFAULT["DIM_PROBLEM"] = 30


def parse_args():
    """
    Parse the argumets provided during program execution. Use the default
    if a given argument is not provided during execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_epochs", type=int, default=DEFAULT["EPOCHS"])
    parser.add_argument("--problem", type=str, default=DEFAULT["PROBLEM"])
    parser.add_argument('--is_penalty_same', type=bool,
                        default=DEFAULT["IS_PENALTY_SAME"])
    parser.add_argument("--dim_hidden", type=int,
                        default=DEFAULT["DIM_HIDDEN"])
    parser.add_argument("--dim_problem", type=int,
                        default=DEFAULT["DIM_PROBLEM"])
    args = parser.parse_args()
    if args.is_penalty_same:
        # The dimension of the context will be 2 more than dimension of problem.
        # We can break down 2 as one representing the total items and the second
        # representing comman penalty for all items.
        args.dim_context = args.dim_problem + 2
    else:
        # The dimension of the context will be 2 times the dimension of problem plus 1.
        # Here we have a separate penalty for each item, hence we multipy dimension of the
        # problem by two and add 1 to representing the total items.
        args.dim_context = 2*args.dim_problem + 1
    # Modify the hidden size based on the problem size, if not provided explicitly
    if args.dim_hidden == DEFAULT["DIM_HIDDEN"]:
        args.dim_hidden = int(args.dim_problem/3)
    return args


def train(args):
    print("Inside Train...")
    reward = []
    scip_reward = []    # scip_opt
    loss_init = []
    generator = instance_generator(args.problem)

    # InitializationPolicy
    init_policy = InitializationPolicy(
        args.dim_problem, args.dim_context, args.dim_hidden)
    # InitializationPolicy Optimizer
    init_opt = T.optim.Adam(init_policy.parameters(), lr=1e-4)

    # Baseline
    baseline_net = Baseline(args.dim_context, args.dim_hidden)
    # Baseline optimizer
    opt_base = T.optim.Adam(baseline_net.parameters(), lr=1e-4)
    # Loss calculator for baseline
    loss_base = T.nn.MSELoss()

    # Train
    for epoch in range(1, args.init_epochs+1):
        print("Epoch : {}".format(epoch))
        # Generate instance
        instance = generator.generate_instance()
        context = instance.get_context()
        # Generate environment
        env = Env_KS(instance, 200)

        # Learn using REINFORCE


        _, tmp_opt,scip_gap = env.extensive_form()
        print("scip is {}".format(tmp_opt))
        print("the gap is {}".format(scip_gap))
        scip_reward.append(tmp_opt)
        reward_, loss_init_ = init_policy.REINFORCE(
            init_opt, env, context)

        reward.append(reward_.item())
        print("reward is {}".format(reward))
        loss_init.append(loss_init_.item())

        if epoch % 50 == 0:
            # Save the data file
            np.save("scip_reward.npy",scip_reward)
            np.save("reward.npy", reward)
            np.save("loss_init.npy", loss_init)
            T.save(init_policy.state_dict(), "init_policy")


if __name__ == "__main__":
    args = parse_args()
    train(args)
