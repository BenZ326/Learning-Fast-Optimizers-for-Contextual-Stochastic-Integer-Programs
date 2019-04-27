import os
import numpy as np
import matplotlib.pyplot as plt


def draw_local_move_stats(folders, gamma, beta, lr_a2c):
    for folder in folders:
        loss = np.load(os.path.join("stats", folder, "loss.npy"))
        rewards = np.load(os.path.join("stats", folder, "rewards.npy"))
        mse = np.load(os.path.join("stats", folder, "mse.npy"))

        total_reward_per_epoch = np.asarray(
            [np.sum(reward) for reward in rewards])
        avg_reward_per_move = np.average(rewards, axis=0)

        num_moves = str(avg_reward_per_move.shape[0])

        # Loss plot
        plt.figure()
        plt.plot(loss, alpha=0.8)
        plt.title("Loss per epoch \n Moves={}, Gamma={}, Beta={}, LR_A2C={}".format(
            num_moves, gamma, beta, lr_a2c))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join("stats", folder, "loss_"+folder+".pdf"))
        plt.close()

        # Reward plot
        plt.figure()
        plt.plot(total_reward_per_epoch, color='m', alpha=0.8)
        plt.title("Total reward or increase in objective value per epoch \n Moves={}, Gamma={}, Beta={}, LR_A2C={}".format(
            num_moves, gamma, beta, lr_a2c))
        plt.xlabel("Epochs")
        plt.ylabel("Total reward")
        plt.savefig(os.path.join("stats", folder, "rpe_"+folder+".pdf"))
        plt.close()

        # Average reward per move
        plt.figure()
        plt.plot(avg_reward_per_move, color='r', alpha=0.8)
        plt.title("Average reward per move \n Moves={}, Gamma={}, Beta={}, LR_A2C={}".format(
            num_moves, gamma, beta, lr_a2c))
        plt.xlabel("Move")
        plt.ylabel("Average reward")
        plt.savefig(os.path.join("stats", folder, "arpm_"+folder+".pdf"))
        plt.close()

        # Mean square error
        plt.figure()
        plt.plot(mse, color='g', marker="*", alpha=0.8)
        plt.title("Mean square error per epoch \n Moves={}, Gamma={}, Beta={}, LR_A2C={}".format(
            num_moves, gamma, beta, lr_a2c))
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.savefig(os.path.join("stats", folder, "mse_"+folder+".pdf"))
        plt.close()


# Find local_move_stats directory
dirs = os.listdir("stats")
local_move_stats_folders = list()
for folder in dirs:
    if "local_move_stats" in folder:
        local_move_stats_folders.append(folder)

num_moves = "50"
gamma = "0.9"
beta = "0.001"
lr_a2c = "0.0001"

tokens = folder.split("_")
for token in tokens:
    if "move=" in token:
        num_moves = token.split("=")[1]
    elif "gamma=" in token:
        gamma = token.split("=")[1]
    elif "beta=" in token:
        beta = token.split("=")[1]
    elif "a2c=" in token:
        lr_a2c = token.split("=")[1]


draw_local_move_stats(local_move_stats_folders, gamma, beta, lr_a2c)
