import argparse

import numpy as np
import pandas as pd
from reinforcementLearner.parsing import parse
from reinforcementLearner.common import player_0_name, player_1_name, gamma
from subprocess import Popen, PIPE
import os

from reinforcementLearner.neural_net import NeuralNet

def main():
    parser = argparse.ArgumentParser(description="Halite II training")
    parser.add_argument("--model_name", help="Name of the model")
    parser.add_argument("--minibatch_size", type=int, help="Size of the minibatch", default=100)
    parser.add_argument("--steps", type=int, help="Number of steps in the training", default=100)
    parser.add_argument("--cache", help="Location of the model we should continue to train")
    parser.add_argument("--games_limit", type=int, help="Train on up to games_limit games", default=1000)
    parser.add_argument("--seed", type=int, help="Random seed to make the training deterministic")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    #neural_net = NeuralNet(cached_model=args.cache)

    gen_data(10)

def gen_data(samples):

    # Min of 2 samples
    if samples == 1:
        samples = 2

    for i in range(samples//2):
        print("Generating samples {} and {}...".format(i*2+1, i*2+2))

        command = ["./halite", "-d", "160 160", "-t", "python3 MyBot.py", "python3 MyBotCPU.py"]

        process = Popen(command, stdout=PIPE)
        out, _ = process.communicate()
        out = out.decode("utf-8")

        # Winner is the index of the player who won
        winner_name = player_0_name
        if out.index("#2 ") < out.index("#1 "):
            winner_name = player_1_name

        # Find the file of the winner
        dir_indices = sorted([int(end[9:-5]) for end in os.listdir("rlData") if end[:9] == "gameData_" and end[-5:] == ".data"], reverse=True)
        f1 = "rlData/gameData_" + str(dir_indices[0]) + ".data"
        f2 = "rlData/gameData_" + str(dir_indices[1]) + ".data"

        clean_data_end(f1)
        clean_data_end(f2)

        with open(f1, "r") as f:
            f1_name = f.readline()[:-1]
        # Write winner and loser
        if f1_name == winner_name:
            with open(f1, "a+") as f:
                f.write("\nWIN")
            with open(f2, "a+") as f:
                f.write("\nLOSS")
        else:
            with open(f1, "a+") as f:
                f.write("\nLOSS")
            with open(f2, "a+") as f:
                f.write("\nWIN")

def discount_rewards(rewards):
    disc_rewards = np.zeros_like(rewards)
    running_reward = 0
    for i in reversed(range(len(rewards))):
        running_reward = running_reward * gamma + rewards[i]
        disc_rewards = running_reward
    return disc_rewards

    print(data_file)
# Get rid of parts left from the last round
# because the last round doesn't finish writing
def clean_data_end(data_file):
    data = ""
    with open(data_file, "r") as f:
        data = f.read()

    del_index = data.rfind("-\n")
    data = data[:del_index+1]

    with open(data_file, "w") as f:
        f.write(data)

if __name__ == "__main__":
    main()
