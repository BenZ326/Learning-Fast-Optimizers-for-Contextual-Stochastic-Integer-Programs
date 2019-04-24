import argparse
import hashlib
import os
import time

import numpy as np
import torch as T
from torch.distributions.bernoulli import Bernoulli

from initialisation_policy import (Baseline, LSTMInitialisationPolicy,
                                   NADEInitializationPolicy,
                                   NNInitialisationPolicy)
from instance import instance_generator
from local_move_policy import A2CLocalMovePolicy

from utils import create_environment
from utils import select_initialization_policy
from utils import update_baseline_model
from utils import generate_dummy_start_state
from utils import evaluate_model
from utils import save_stats_and_model
from utils import save_local_move_policy_stats_and_model

##################################################################
#                       DEFAULT PARAMETERS                       #
##################################################################
NADE = "NADE"
FFNN = "FFNN"
LSTM = "LSTM"
JOINT = "JOINT"
INIT = "INIT"
LOCAL = "LOCAL"

DEFAULT = {}
DEFAULT["EPOCHS"] = 500
DEFAULT["PROBLEM"] = "ks"
DEFAULT["IS_PENALTY_SAME"] = True
DEFAULT["DIM_HIDDEN"] = 10
DEFAULT["DIM_PROBLEM"] = 25
DEFAULT["INIT_MODEL"] = NADE
DEFAULT["INIT_LR_RATE"] = 1e-4
DEFAULT["NUM_OF_SCENARIOS"] = 200
DEFAULT["USE_BASELINE"] = False
DEFAULT["WINDOW_SIZE"] = 5
DEFAULT["NUM_OF_SCENARIOS_IN_STATE"] = 40
DEFAULT["TRAIN_MODE"] = LOCAL


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
    parser.add_argument("--window_size", type=int,
                        default=DEFAULT["WINDOW_SIZE"])
    parser.add_argument("--num_of_scenarios_in_state", type=int,
                        default=DEFAULT["NUM_OF_SCENARIOS_IN_STATE"])
    parser.add_argument("--train_mode", type=str,
                        default=DEFAULT["TRAIN_MODE"])

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


def train_joint(args):
    """
    Simultaneously train Initialisation Policy and Baseline

    Arguments
    ---------
    args : dict
        Dictionary containing command line arguments


    """
    # Buffers to store statistics
    reward = list()
    loss_init = list()
    mean_square_error_per_epoch = list()
    mean_relative_distance_per_epoch = list()

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

        env = create_environment(args, instance)

        # Learn using REINFORCE
        # If using baseline, update the baseline net
        if args.use_baseline:
            baseline_reward = baseline_net.forward(context)
            reward_, loss_init_, start_state = init_policy.REINFORCE(
                init_opt, env, context, baseline_reward, True)
            update_baseline_model(
                loss_base_fn, baseline_reward, reward_, opt_base)
        # Without using baseline
        else:
            reward_, loss_init_, start_state = init_policy.REINFORCE(
                init_opt, env, context)

        local_move_policy.train(start_state, env)

        reward.append(reward_.item())
        loss_init.append(loss_init_.item())

        # Save stats and model
        if epoch % 50 == 0:
            eval_stats = evaluate_model(
                args, env, generator, init_policy=init_policy)

            mean_square_error_per_epoch.append(eval_stats["mean_square_error"])
            mean_relative_distance_per_epoch.append(
                eval_stats["mean_relative_distance"])

            save_stats_and_model(args, init_policy, reward, loss_init, ev_scip, ev_policy,
                                 ev_gap, ev_random, eval_sqdist, eval_rp, eval_nbr)


def train_local_move_policy(args):
    """
    Train Local Move Policy only

    Arguments
    ---------
    args : dict
        Dictionary containing command line arguments
    """
    rewards_per_epoch = list()
    loss_per_epoch = list()
    mean_square_error_per_epoch = list()
    mean_relative_distance_per_epoch = list()

    # Instance generator
    generator = instance_generator(args.problem)

    # Initialise local move policy
    local_move_policy = A2CLocalMovePolicy(
        args, gamma=0.1, beta_entropy=1e-3, num_local_move=50)

    # Train
    for epoch in range(1, args.init_epochs+1):
        start_time = time.time()
        print("******************************************************")
        print(f"Epoch : {epoch}")

        # Generate instance and environment
        instance = generator.generate_instance()
        context = instance.get_context()
        env = create_environment(args, instance)

        start_state = generate_dummy_start_state(env, args.dim_problem)

        # Take num_local_moves to improves the provided initial solution
        rewards, loss = local_move_policy.train(start_state, env)

        rewards_per_epoch.append(rewards)
        loss_per_epoch.append(loss)

        # Save stats and model
        if epoch % 100 == 0:
            eval_stats = evaluate_model(
                args, env, generator, local_move_policy=local_move_policy)
            mean_square_error_per_epoch.append(eval_stats["mean_square_error"])
            mean_relative_distance_per_epoch.append(
                eval_stats["mean_relative_distance"])

            save_local_move_policy_stats_and_model(args,
                                                   local_move_policy,
                                                   epoch,
                                                   rewards_per_epoch,
                                                   loss_per_epoch,
                                                   mean_square_error_per_epoch,
                                                   mean_relative_distance_per_epoch)

        print(
            f"Took {time.time() - start_time} in epoch {epoch}/{args.init_epochs}")


if __name__ == "__main__":
    args = parse_args()
    if args.train_mode == JOINT:
        train_joint(args)
    elif args.train_mode == INIT:
        print("Independent training of init policy not yet defined.")
    elif args.train_mode == LOCAL:
        train_local_move_policy(args)
