from environment import Env_KS
from instance import instance_generator
from initialisation_policy import NNInitialisationPolicy
from initialisation_policy import LSTMInitialisationPolicy
from initialisation_policy import NADEInitializationPolicy
from initialisation_policy import Baseline
from state import state
from local_move_policy import A2CLocalMovePolicy


import torch as T
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import argparse
import os
import hashlib
import time

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
DEFAULT["NUM_OF_SCENARIOS"] = 200
DEFAULT["USE_BASELINE"] = False
DEFAULT["WINDOW_SIZE"] = 5
DEFAULT["NUM_OF_SCENARIOS_IN_STATE"] = 40


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
                        default=DEFAULT["NUM_OF_SCENARIOS"])

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

    parser.add_argument("--window_size", type=int,
                        default=DEFAULT["WINDOW_SIZE"])
    parser.add_argument("--num_of_scenarios_in_state", type=int,
                        default=DEFAULT["NUM_OF_SCENARIOS_IN_STATE"])

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
    """
    Update Baseline Policy by minimizing its loss

    Arguments
    ---------
    loss_fn : nn.MSELoss()
        Object of MSELoss
    y_hat : Tensor
        Prediction reward by Baseline model
    y : Tensor
        True reward
    optimizer : T.optim.Adam()
        Object of Adam optimizer
    """
    loss = loss_fn(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate_model(args, model, generator, eval_sqdist, eval_rp, ev_random,
                   eval_nbr,  ev_scip, ev_policy, ev_gap, TEST_INSTANCES=10):
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


def save_stats_and_model(args, init_policy, reward, loss_init, ev_scip, ev_policy,
                         ev_gap, ev_random, eval_sqdist, eval_rp, eval_nbr):
    """
    Save statistics and model 

    Arguments
    ---------
    args : dict
        Dictionary containing command line arguments
    init_policy : nn.Module
        Object of Initialisation policy
    reward : list 
        a list that stores reward return by the initialization policy at each epoch
    loss_init : list 
        a list that stores loss at each epoch    
    ev_scip : list 
        a list that stores the optimal objective values returned by SCIP
    ev_policy : list
        a list that stores the values returned by the initialization policy
    ev_gap : list 
        a list that stores MIP gap when SCIP is terminated
    ev_random : list
        a list that stores the values returned by random solutions
    eval_sqdist : list
        a list that stores square distance between scip solver's result and the policy's
    eval_rp : list 
        a list that stores relative percentages
    eval_nbr : list
        a list that stores how many times the policy works better than SCIP
    """

    if not os.path.exists("stats"):
        os.mkdir("stats")

    prefix = "_".join(
        [args.init_model,
            "baseline="+str(args.use_baseline),
            "lr="+str(args.init_lr_rate),
            "epochs="+str(args.init_epochs)])

    if not os.path.exists(os.path.join("stats", prefix)):
        os.mkdir(os.path.join("stats", prefix))

    np.save(os.path.join("stats", prefix, "reward.npy"), reward)
    np.save(os.path.join("stats", prefix, "loss_init.npy"), loss_init)
    np.save(os.path.join("stats", prefix, "ev_random.npy"), ev_random)
    np.save(os.path.join("stats", prefix, "ev_scip.npy"), ev_scip)
    np.save(os.path.join("stats", prefix, "ev_policy"), ev_policy)
    np.save(os.path.join("stats", prefix, "env_gap"), env_gap)
    np.save(os.path.join("stats", prefix,
                         "eval_sqdist.npy"), eval_sqdist)
    np.save(os.path.join("stats", prefix, "eval_rp.npy"), eval_rp)
    np.save(os.path.join("stats", prefix, "eval_nbr.npy"), eval_nbr)

    T.save(init_policy.state_dict(), os.path.join(
        "stats", prefix, "init_policy"))


def select_initialization_policy(args):
    """ 
    Selects initialisation policy based on the command line argument

    Arguments
    ---------
    args : dict
        Dictionary containing command line arguments

    Returns
    -------
    init_policy : nn.module
        Object of Initialisation policy
    """
    if args.init_model == STR["NADE"]:
        # Use NADE as initialisation policy
        init_policy = NADEInitializationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)
    elif args.init_model == STR["FFNN"]:
        # Use FFNN as initialisation policy
        init_policy = NNInitialisationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)
    elif args.init_model == STR["LSTM"]:
        # Use FFNN as initialisation policy
        init_policy = LSTMInitialisationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)

    return init_policy


def train(args):
    """
    Simultaneously train Initialisation Policy and Baseline

    Arguments
    ---------
    args : dict
        Dictionary containing command line arguments


    """
    # Buffers to store statistics
    reward, loss_init, ev_scip, ev_policy, ev_gap, ev_random, eval_sqdist,
    eval_rp, eval_nbr = list(), list(), list(
    ), list(), list(), list(), list(), list(), list()

    generator = instance_generator(args.problem)

    # Initialise Initialisation policy and set its optimizer
    init_policy = select_initialization_policy(args)
    init_opt = T.optim.Adam(init_policy.parameters(), lr=args.init_lr_rate)

    # Initialize Baseline if required
    if args.use_baseline:
        baseline_net = Baseline(args.dim_context, args.dim_hidden)
        opt_base = T.optim.Adam(baseline_net.parameters(), lr=1e-4)
        loss_base_fn = T.nn.MSELoss()

    # Initialise local move policy
    local_move_policy = A2CLocalMovePolicy(args)

    # Train
    for epoch in range(1, args.init_epochs+1):
        print("******************************************************")
        print(f"Epoch : {epoch}")
        # Generate instance and environment
        instance = generator.generate_instance()
        context = instance.get_context()
        env = Env_KS(instance, args.num_of_scenarios)

        # Learn using REINFORCE
        # If using baseline, update the baseline net
        # if args.use_baseline:
        #     baseline_reward = baseline_net.forward(context)
        #     reward_, loss_init_ = init_policy.REINFORCE(
        #         init_opt, env, context, baseline_reward, True)
        #     update_baseline_model(
        #         loss_base_fn, baseline_reward, reward_, opt_base)
        # # Without using baseline
        # else:
        #     reward_, loss_init_ = init_policy.REINFORCE(
        #         init_opt, env, context)

        local_move_policy.train(env)

        reward.append(reward_.item())
        loss_init.append(loss_init_.item())

        # Save stats and model
        if epoch % 50 == 0:
            evaluate_model(args, init_policy, generator, eval_sqdist, eval_rp,
                           ev_random, eval_nbr, ev_scip, ev_policy, env_gap)
            save_stats(args, init_policy, reward, loss_init, ev_scip, ev_policy,
                       ev_gap, ev_random, eval_sqdist, eval_rp, eval_nbr)


if __name__ == "__main__":
    args = parse_args()
    train(args)
