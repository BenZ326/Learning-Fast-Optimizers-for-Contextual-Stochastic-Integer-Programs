import time
import os

import numpy as np

from environment import EnvKnapsack

STR = {}
STR["JOINT"] = "JOINT"
STR["INIT"] = "INIT"
STR["LOCAL"] = "LOCAL"


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

    if args.train_mode == STR["INIT"]:
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
        if args.train_mode == STR["LOCAL"]:
            # Improve the solution using local move policy only
            start_state = generate_dummy_start_state(env, args.dim_problem)
            solution = local_move_policy.perform_local_moves_in_eval_mode(
                env, start_state)
        elif args.train_mode == STR["INIT"]:
            # Solve instance using initialisation policy only
            solution, _ = init_policy.forward(context)
        elif args.train_mode == STR["JOINT"]:
            # Solve instance using initialisation policy only
            solution, _ = init_policy.forward(context)
            state, _, _, _ = env.step(solution=solution)
            solution = local_move_policy.perform_local_moves_in_eval_mode(
                env, state)

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
            "epochs="+str(args.epochs)])

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


def save_local_move_policy_stats_and_model(args,
                                           local_move_policy,
                                           epoch,
                                           rewards,
                                           loss,
                                           mean_square_error,
                                           mean_relative_distance):
    """
    Save the statistics of the local move policy

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

    prefix = "_".join(
        ["local_move_stats",
         "train_mode="+args.train_mode,
         "epochs="+str(args.epochs),
         "local_move="+str(local_move_policy.num_local_move),
         "gamma="+str(local_move_policy.gamma),
         "beta="+str(local_move_policy.beta_entropy),
         "lr_a2c="+str(local_move_policy.lr_a2c)
         ])

    if not os.path.exists(os.path.join("stats", prefix)):
        os.mkdir(os.path.join("stats", prefix))

    np.save(os.path.join("stats", prefix, "rewards.npy"), np.asarray(rewards))
    np.save(os.path.join("stats", prefix, "loss.npy"), np.asarray(loss))
    np.save(os.path.join("stats", prefix, "mse.npy"),
            np.asarray(mean_square_error))
    np.save(os.path.join("stats", prefix, "mrd.npy"),
            np.asarray(mean_relative_distance))

    local_move_policy.save_model(os.path.join("stats", prefix))
