from environment import Env_KS
from instance import instance_generator
from initialisation_policy import NNInitialisationPolicy
from initialisation_policy import LSTMInitialisationPolicy
from initialisation_policy import NADEInitializationPolicy
from initialisation_policy import Baseline

import torch as T
from torch.distributions.bernoulli import Bernoulli
import argparse
import numpy as np
import os

##################################################################
#                       DEFAULT PARAMETERS                       #
##################################################################
DEFAULT = {}
DEFAULT["EPOCHS"] = 500
DEFAULT["PROBLEM"] = "ks"
DEFAULT["IS_PENALTY_SAME"] = True
DEFAULT["DIM_HIDDEN"] = 10
DEFAULT["DIM_PROBLEM"] = 25
DEFAULT["INIT_MODEL"] = "NADE"
DEFAULT["INIT_LR_RATE"] = 1e-4
DEFAULT["NO_OF_SCENARIOS"] = 200
DEFAULT["USE_BASELINE"] = False


STR = {}
STR["NADE"] = "NADE"
STR["FFNN"] = "FFNN"
STR["LSTM"] = "LSTM"

N_w = 200

# folders
DEFAULT["DATA_NADE"] = "data_nade/"
DEFAULT["DATA_NN"] = "data_nn/"
DEFAULT["DATA_LSTM"] = "data_lstm/"


def parse_args():
    """
    Parse the argumets provided during program execution. Use the default
    if a given argument is not provided during execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str,
                        default=DEFAULT["INIT_MODEL"])
    parser.add_argument("--init_lr_rate", type=float,
                        default=DEFAULT["INIT_LR_RATE"])
    parser.add_argument("--init_epochs", type=int, default=DEFAULT["EPOCHS"])
    parser.add_argument("--use_baseline", type=bool,
                        default=DEFAULT["USE_BASELINE"])
    parser.add_argument('--is_penalty_same', type=bool,
                        default=DEFAULT["IS_PENALTY_SAME"])
    parser.add_argument("--num_of_scenarios", type=int,
                        default=DEFAULT["NO_OF_SCENARIOS"])

    parser.add_argument("--problem", type=str, default=DEFAULT["PROBLEM"])
    parser.add_argument("--dim_hidden", type=int,
                        default=DEFAULT["DIM_HIDDEN"])
    parser.add_argument("--dim_problem", type=int,
                        default=DEFAULT["DIM_PROBLEM"])
    parser.add_argument("--path_nade", type=str,
                        default=DEFAULT["DATA_NADE"])

    parser.add_argument("--path_nn", type=str,
                        default=DEFAULT["DATA_NN"])
    parser.add_argument("--path_lstm", type=str,
                        default=DEFAULT["DATA_NADE"])

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


def update_baseline_model(loss_fn, y_hat, y, optimizer):
    """Update the loss of the Baseline Policy
    """
    loss = loss_fn(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate_model(args, model, generator, eval_sqdist, eval_rp, ev_random, eval_nbr,  ev_scip, ev_policy, ev_gap, TEST_INSTANCES=10):
    """
    The evaulation function to compare the gap between initialization policy and scip solver
    policy: the policy
    generator: instance generator
    """
    print("starting to evaluate")
    square_dist = 0
    relative_dist = 0
    num_rl_better = 0
    model.eval()
    for i in range(TEST_INSTANCES):
        # Solve instance using SCIP
        instance = generator.generate_instance()
        context = instance.get_context()
        env = Env_KS(instance, N_w)
        _, reward_scip, scip_gap = env.extensive_form()
        ev_gap.append(scip_gap)
        ev_scip.append(reward_scip)

        # Solve instance using LM solver
        solution, _ = model.forward(context)

        reward_policy = env.step(solution.numpy().reshape(-1))[0]
        ev_policy.append(reward_policy)

        # Generate random vectors to compare the result
        random_solution = np.random.randint(
            0, 2, size=args.dim_problem).reshape(-1)
        reward_random = env.step(random_solution)[0]
        ev_random.append(reward_random)

        # Calculate stats about performance. How close we are to the
        # scip
        square_dist += (reward_scip - reward_policy) ** 2
        relative_dist += (reward_scip - reward_policy) / reward_scip
        print("reward got by scip is {}".format(reward_scip))
        print("reward got by policy is {}".format(reward_policy))
        print("reward got by random is {}".format(reward_random))
        if reward_scip <= reward_policy:
            num_rl_better += 1

    eval_sqdist.append(square_dist / TEST_INSTANCES)
    eval_rp.append(relative_dist / TEST_INSTANCES)
    eval_nbr.append(num_rl_better)


def train(args):
    print("Inside Train...")
    reward = []
    loss_init = []
    ev_scip = []
    ev_policy = []
    ev_gap = []
    ev_random = []
    generator = instance_generator(args.problem)
    folder_path = None
    name_base_line = ""
    # Select initialisation policy
    if args.init_model == STR["NADE"]:
        # Use NADE as initialisation policy
        init_policy = NADEInitializationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)
        folder_path = args.path_nade
    elif args.init_model == STR["FFNN"]:
        # Use FFNN as initialisation policy
        init_policy = NNInitialisationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)
        folder_path = args.path_nn
    elif args.init_model == STR["LSTM"]:
        # Use FFNN as initialisation policy
        init_policy = LSTMInitialisationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)
        folder_path = args.path_lstm
    os.mkdir(folder_path)
    init_opt = T.optim.Adam(init_policy.parameters(), lr=args.init_lr_rate)

    # Initialize baseline if required
    if args.use_baseline:
        baseline_net = Baseline(args.dim_context, args.dim_hidden)
        opt_base = T.optim.Adam(baseline_net.parameters(), lr=1e-4)
        loss_base_fn = T.nn.MSELoss()
        name_base_line = "base_line"
    eval_sqdist = []
    eval_rp = []
    eval_nbr = []     # record number of RL better

    # Train
    for epoch in range(1, args.init_epochs+1):
        # if epoch == 1:
        #     evaluate_model(args, init_policy, generator, eval_sqdist,
        #                    eval_rp, ev_random, eval_nbr, ev_scip, ev_policy, ev_gap)
        print("******************************************************")
        print("Epoch : {}".format(epoch))
        # Generate instance and environment
        instance = generator.generate_instance()
        context = instance.get_context()
        env = Env_KS(instance, DEFAULT["NO_OF_SCENARIOS"])

        # Learn using REINFORCE
        # If using baseline, update the baseline net
        if args.use_baseline:
            baseline_reward = baseline_net.forward(context)
            reward_, loss_init_ = init_policy.REINFORCE(
                init_opt, env, context, baseline_reward, True)
            update_baseline_model(
                loss_base_fn, baseline_reward, reward_, opt_base)
        # Without using baseline
        else:
            reward_, loss_init_ = init_policy.REINFORCE(
                init_opt, env, context)

        reward.append(reward_.item())
        loss_init.append(loss_init_.item())
        if epoch % 50 == 0:
            # Save the data file
            evaluate_model(args, init_policy, generator, eval_sqdist, eval_rp, ev_random, eval_nbr, ev_scip, ev_policy,
                           ev_gap)
            folder_path = args.path_nade
            if args.use_baseline:
                folder_path = args.path_nade_baseline
            np.save(file_name_prifix(folder_path,name_base_line)+"ev_random.npy", ev_random)
            """
            Explanation of statistics variables: 
            All the following variables are only associated with "evaluation":
            ev_scip: a list that stores the optimal objective values returned by SCIP
            ev_policy: a list that stores the values returned by the initialization policy
            ev_gap: a list that stores MIP gap when SCIP is terminated
            eval_sqdist: a list that stores square distance between scip solver's result and the policy's
            eval_rp: a list that stores relative percentages
            eval_nbr: a list that stores how many times a the policy works better than SCIP
            ------------------------------------------------------------------------------
            reward: a list that stores reward return by the initialization policy at each epoch
            loss_init: a list that stores loss at each epoch
            """
            np.save(folder_path+"ev_scip.npy", ev_scip)
            np.save(folder_path+"ev_policy", ev_policy)
            np.save(folder_path+"ev_gap", ev_gap)
            np.save(folder_path+"eval_sqdist.npy", eval_sqdist)
            np.save(folder_path+"eval_rp.npy", eval_rp)
            np.save(folder_path+"eval_nbr.npy", eval_nbr)
            np.save(folder_path+"reward.npy", reward)
            np.save(folder_path+"loss_init.npy", loss_init)
            T.save(init_policy.state_dict(), folder_path+"init_policy")

def file_name_prifix(folder, base_line):
    return folder+base_line+"_"

if __name__ == "__main__":
    args = parse_args()
    train(args)
