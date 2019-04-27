import time
import os
import torch as T
import numpy as np

from environment import EnvKnapsack

from initialisation_policy import (LSTMInitialisationPolicy,
                                   NADEInitializationPolicy,
                                   NNInitialisationPolicy)

JOINT = "JOINT"
INIT = "INIT"
LOCAL = "LOCAL"
NADE = "NADE"
LSTM = "LSTM"
FFNN = "FFNN"


def create_environment(args, instance):
    """
    Create environment based on the problem definition

    Arguments
    ---------
    problem_type : str
        The problem type for which we are creating the envrionment
    instance : ndarray
        An instance of the problem

    Returns
    -------
    env : Object <class problem_type>
        Returns an environment object based on the problem_type
    """
    if args.problem == "ks":
        from environment import EnvKnapsack
        env = EnvKnapsack(args, instance)

        return env


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
    if args.init_model == NADE:
        # Use NADE as initialisation policy
        init_policy = NADEInitializationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)
    elif args.init_model == FFNN:
        # Use FFNN as initialisation policy
        init_policy = NNInitialisationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)
    elif args.init_model == LSTM:
        # Use FFNN as initialisation policy
        init_policy = LSTMInitialisationPolicy(
            args.dim_problem, args.dim_context, args.dim_hidden)

    return init_policy


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


def generate_dummy_start_state(env, dim_problem):
    """
    Generate a dummy start state

    Return
    ------
    start_state : Object <class State>
        A dummy start state
    """
    solution = np.random.choice(
        [0, 1], size=(dim_problem,), p=[0.5, 0.5])
    start_state, _ = env.step(solution=solution.reshape(-1))

    return start_state


def evaluate_model(args, env, generator, init_policy=None, local_move_policy=None, TEST_INSTANCES=10):
    """
    Evaluate the performance of local move policy.

    args : dict
        Dictionary of arguments
    """
    square_dist = 0
    relative_dist = 0

    if args.train_mode == INIT:
        init_policy.eval()

    print("#######################################")
    for i in range(TEST_INSTANCES):
        print(f"---> Evaluating scenario {i+1}...")

        # Solve instance using SCIP
        start_time = time.time()
        instance = generator.generate_instance()
        context = instance.get_context()
        env = EnvKnapsack(args, instance)
        _, reward_scip, _, _ = env.extensive_form()
        print(f"Took {time.time() - start_time} for extensive form")

        # Solve using model
        start_time = time.time()
        if args.train_mode == LOCAL:
            # Improve the solution using local move policy only
            start_state = generate_dummy_start_state(env, args.dim_problem)
            solution = local_move_policy.perform_local_moves_in_eval_mode(
                env, start_state)
            # solution = solution.numpy()
        elif args.train_mode == INIT:
            # Solve instance using initialisation policy only
            solution, _ = init_policy.forward(context)
            solution = solution.numpy()
        elif args.train_mode == JOINT:
            # Solve instance using initialisation policy only
            solution, _ = init_policy.forward(context)
            solution = solution.numpy().reshape(-1)
            state, _ = env.step(solution=solution)
            solution = local_move_policy.perform_local_moves_in_eval_mode(
                env, state)
            # solution = solution.numpy()

        # Get the reward corresponding to the improved solution
        _, reward_policy = env.step(solution=solution.reshape(-1))
        print(f"Took {time.time() - start_time} for Model")

        # Generate random vectors to compare the result
        start_time = time.time()
        random_solution = np.random.randint(
            0, 2, size=args.dim_problem).reshape(-1)
        _, reward_random = env.step(solution=random_solution)
        print(f"Took {time.time() - start_time} for random")

        # Calculate stats about performance. How close we are to the
        # scip
        square_dist += (reward_scip - reward_policy) ** 2
        relative_dist += (reward_scip - reward_policy) / reward_scip
        print("reward got by scip is {}".format(reward_scip))
        print("reward got by policy is {}".format(reward_policy))
        print("reward got by random is {}".format(reward_random))

    stats = {
        "mean_square_error": square_dist / TEST_INSTANCES,
        "mean_relative_distance": relative_dist / TEST_INSTANCES
    }

    return stats


def save_stats_and_model(args,
                         epoch,
                         rewards,
                         loss,
                         mean_square_error,
                         mean_relative_distance,
                         model,
                         model_type):
    """
    Save the statistics and model

    Arguments
    ---------

    args : dict
        Dictionary of arguments
    local_move_policy : Object <class A2CLocalMovePolicy>
        Object of local move policy
    epoch : int
        Epoch after which the data is saved
    rewards : list
        A list of lists containing rewards obtained for each local move
        for a given epoch
    loss : list
        A list containing total loss per epoch 
    mean_square_error : list
        A list containing MSE between SCIP and Local move solver objective value
    mean_relative_distance : list
        A list containing Mean relative distance between SCIP and Local move 
        solver objective value    
    """

    if not os.path.exists("stats"):
        os.mkdir("stats")

    if model_type == INIT:
        prefix = "_".join(
            ["init_stats",
             "train_mode="+args.train_mode,
             "epochs="+str(args.epochs),
             "baseline="+str(args.use_baseline),
             "lr_init="+str(args.init_lr_rate)
             ])

    elif model_type == LOCAL:
        prefix = "_".join(
            ["local_move_stats",
             "train_mode="+args.train_mode,
             "epochs="+str(args.epochs),
             "local_move="+str(args.num_local_move),
             "gamma="+str(model.gamma),
             "beta="+str(model.beta_entropy),
             "lr_a2c="+str(model.lr_a2c)
             ])

    if not os.path.exists(os.path.join("stats", prefix)):
        os.mkdir(os.path.join("stats", prefix))

    np.save(os.path.join("stats", prefix, "rewards.npy"), np.asarray(rewards))
    np.save(os.path.join("stats", prefix, "loss.npy"), np.asarray(loss))
    np.save(os.path.join("stats", prefix, "mse.npy"),
            np.asarray(mean_square_error))
    np.save(os.path.join("stats", prefix, "mrd.npy"),
            np.asarray(mean_relative_distance))

    if model_type == LOCAL:
        model.save_model(os.path.join("stats", prefix))
    else:
        T.save(model.state_dict(), os.path.join(
            "stats", prefix, "init_policy.pt"))
