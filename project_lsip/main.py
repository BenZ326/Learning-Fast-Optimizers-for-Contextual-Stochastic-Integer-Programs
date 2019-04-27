import argparse
import hashlib
import os
import time

import numpy as np
import torch as T
from torch.distributions.bernoulli import Bernoulli

from initialisation_policy import Baseline
from instance import instance_generator
from local_move_policy import A2CLocalMovePolicy

from utils import create_environment
from utils import select_initialization_policy
from utils import update_baseline_model
from utils import generate_dummy_start_state
from utils import evaluate_model
from utils import save_stats_and_model


##################################################################
#                       DEFAULT PARAMETERS                       #
##################################################################
NADE = "NADE"
FFNN = "FFNN"
LSTM = "LSTM"
JOINT = "JOINT"
INIT = "INIT"
LOCAL = "LOCAL"
GAMMA = 0.9
BETA = 1e-3
NUM_OF_LOCAL_MOVE = 50
EPOCHS = 500
PROBLEM = "ks"
IS_PENALTY_SAME = True
DIM_HIDDEN = 10
DIM_PROBLEM = 25
INIT_LR_RATE = 1e-4
NUM_OF_SCENARIOS_FOR_EXPECTATION = 200
USE_BASELINE = False
WINDOW_SIZE = 5
NUM_OF_SCENARIOS_IN_STATE = 40
TRAIN_MODE = LOCAL
LR_A2C = 1e-4

INIT_MODEL = NADE


def parse_args():
    """
    Parse the argumets provided during program execution. Use the default
    if a given argument is not provided during execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str,
                        default=INIT_MODEL)
    parser.add_argument("--init_lr_rate", type=float,
                        default=INIT_LR_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--use_baseline", type=bool,
                        default=USE_BASELINE)
    parser.add_argument('--is_penalty_same', type=bool,
                        default=IS_PENALTY_SAME)
    parser.add_argument("--num_of_scenarios_for_expectation", type=int,
                        default=NUM_OF_SCENARIOS_FOR_EXPECTATION)
    parser.add_argument("--problem", type=str, default=PROBLEM)
    parser.add_argument("--dim_hidden", type=int,
                        default=DIM_HIDDEN)
    parser.add_argument("--dim_problem", type=int,
                        default=DIM_PROBLEM)
    parser.add_argument("--window_size", type=int,
                        default=WINDOW_SIZE)
    parser.add_argument("--num_of_scenarios_in_state", type=int,
                        default=NUM_OF_SCENARIOS_IN_STATE)
    parser.add_argument("--train_mode", type=str,
                        default=TRAIN_MODE)

    parser.add_argument("--gamma", type=float,
                        default=GAMMA)
    parser.add_argument("--beta", type=float,
                        default=BETA)
    parser.add_argument("--num_local_move", type=int,
                        default=NUM_OF_LOCAL_MOVE)
    parser.add_argument("--lr_a2c", type=float, default=LR_A2C)
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
    if args.dim_hidden == DIM_HIDDEN:
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
    reward_init_buffer = list()
    loss_init_buffer = list()

    rewards_local_buffer = list()
    loss_local_buffer = list()

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
    local_move_policy = A2CLocalMovePolicy(args.dim_context, args.dim_problem, args.window_size,
                                           args.num_of_scenarios_in_state, gamma=args.gamma,
                                           beta_entropy=args.beta, num_local_move=args.num_local_move)

    # Train
    for epoch in range(1, args.epochs+1):
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
            reward_init, loss_init, start_state = init_policy.REINFORCE(
                init_opt, env, context, baseline_reward, True)
            update_baseline_model(
                loss_base_fn, baseline_reward, reward_init, opt_base)
        # Without using baseline
        else:
            reward_init, loss_init, start_state = init_policy.REINFORCE(
                init_opt, env, context)
        reward_init_buffer.append(reward_init)
        loss_init_buffer.append(loss_init)

        # Learn using A2C
        rewards_local, loss_local = local_move_policy.train(start_state, env)
        rewards_local_buffer.append(rewards_local)
        loss_local_buffer.append(loss_local)

        # Save stats and model
        if epoch % 100 == 0:
            eval_stats = evaluate_model(
                args, env, generator, init_policy=init_policy, local_move_policy=local_move_policy)

            mean_square_error_per_epoch.append(eval_stats["mean_square_error"])
            mean_relative_distance_per_epoch.append(
                eval_stats["mean_relative_distance"])

            # Save init policy stats
            save_stats_and_model(args,
                                 epoch,
                                 reward_init_buffer,
                                 loss_init_buffer,
                                 mean_square_error_per_epoch,
                                 mean_relative_distance_per_epoch,
                                 init_policy,
                                 INIT)

            # Save local move policy stats
            save_stats_and_model(args,
                                 epoch,
                                 rewards_local_buffer,
                                 loss_local_buffer,
                                 mean_square_error_per_epoch,
                                 mean_relative_distance_per_epoch,
                                 local_move_policy,
                                 LOCAL)


def train_init_policy(args):
    """
    Train the Intialisation Policy

    Arguments
    ---------
    args : dict
        Dictionary containing command line arguments
    """
    rewards_per_epoch = list()
    loss_per_epoch = list()
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

    # Train
    for epoch in range(1, args.epochs+1):
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

        rewards_per_epoch.append(reward_.item())
        loss_per_epoch.append(loss_init_.item())

        # Save stats and model
        if epoch % 50 == 0:
            eval_stats = evaluate_model(
                args, env, generator, init_policy=init_policy)

            mean_square_error_per_epoch.append(eval_stats["mean_square_error"])
            mean_relative_distance_per_epoch.append(
                eval_stats["mean_relative_distance"])

            # Save init policy stats
            save_stats_and_model(args,
                                 epoch,
                                 rewards_per_epoch,
                                 loss_per_epoch,
                                 mean_square_error_per_epoch,
                                 mean_relative_distance_per_epoch,
                                 init_policy,
                                 INIT)


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
    local_move_policy = A2CLocalMovePolicy(args.dim_context, args.dim_problem, args.window_size,
                                           args.num_of_scenarios_in_state, gamma=args.gamma,
                                           beta_entropy=args.beta, num_local_move=args.num_local_move,
                                           lr_a2c=args.lr_a2c)

    # Train
    for epoch in range(1, args.epochs+1):
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

            save_stats_and_model(args,
                                 epoch,
                                 rewards_per_epoch,
                                 loss_per_epoch,
                                 mean_square_error_per_epoch,
                                 mean_relative_distance_per_epoch,
                                 local_move_policy,
                                 LOCAL)

        print(
            f"Took {time.time() - start_time} in epoch {epoch}/{args.epochs}")


if __name__ == "__main__":
    args = parse_args()
    if args.train_mode == JOINT:
        train_joint(args)
    elif args.train_mode == INIT:
        train_init_policy(args)
    elif args.train_mode == LOCAL:
        train_local_move_policy(args)
