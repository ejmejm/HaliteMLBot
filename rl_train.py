import argparse

import numpy as np
import pandas as pd
from reinforcementLearner.parsing import parse
from reinforcementLearner.common import player_0_name, player_1_name, gamma, PLANET_MAX_NUM, PER_PLANET_FEATURES
from subprocess import Popen, PIPE
import os
import sys

from reinforcementLearner.neural_net import NeuralNet

def main():
    parser = argparse.ArgumentParser(description="Halite II training")
    parser.add_argument("--model_name", help="Name of the model")
    parser.add_argument("--minibatch_size", type=int, help="Size of the minibatch", default=100)
    parser.add_argument("--steps", type=int, help="Number of steps in the training", default=3000)
    parser.add_argument("--cache", help="Location of the model we should continue to train")
    parser.add_argument("--games_limit", type=int, help="Train on up to games_limit games", default=1000)
    parser.add_argument("--sp_batch_size", type=int, help="How many self play games to generate per round of training", default=20)
    parser.add_argument("--seed", type=int, help="Random seed to make the training deterministic")
    parser.add_argument("--cpu", type=bool, help="Use the CPU for training or not", default=False)

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    for i in range(0, args.games_limit, args.sp_batch_size):
        print("Cleaning old data...")
        del_sp_data()

        print("Generating new self play data...")
        gen_data(args.sp_batch_size, cache=args.cache, debug=False)

        if args.cache == "None" or args.cache is None:
            cached_model = None
        else:
            cached_model = "models/" + args.cache

        if not args.cpu:
            nn = NeuralNet(cached_model=cached_model, seed=args.seed, processor="CPU")
        else:
            nn = NeuralNet(cached_model=cached_model, seed=args.seed, processor="GPU")

        x_data, actions, rewards = read_data()
        # rewards = stack_rewards(rewards)
        print("Epoch avg reward:", np.average(rewards))

        data_size = len(x_data)
        x_train, actions_train, rewards_train = x_data[:int(0.85 * data_size)], actions[:int(0.85 * data_size)], rewards[:int(0.85 * data_size)]
        x_validation, actions_validation, rewards_validation = x_data[int(0.85 * data_size):], actions[int(0.85 * data_size):], rewards[int(0.85 * data_size):]

        training_data_size = len(x_train)

        # randomly permute the data
        permutation = np.random.permutation(training_data_size)
        x_train, actions_train, rewards_train = x_train[permutation], actions_train[permutation], rewards_train[permutation]
        print("Initial, cross validation loss: {}".format(nn.compute_loss(x_validation, actions_validation, rewards_validation)))

        curves = []

        print("Begin training epoch")
        for s in range(args.steps):
            start = (s * args.minibatch_size) % training_data_size
            end = start + args.minibatch_size
            training_loss = nn.fit(x_train[start:end], actions_train[start:end], rewards_train[start:end])
            print("Batch avg reward:", np.average(rewards_train[start:end]))
            if s % 25 == 0 or s == args.steps - 1:
                validation_loss = nn.compute_loss(x_validation, actions_validation, rewards_validation)
                #print("Step: {}, cross validation loss: {}, training_loss: {}".format(s, validation_loss, training_loss))
                # print(nn.debug_stat(x_validation, actions_validation, rewards_validation))
                curves.append((s, training_loss, validation_loss))

        cf = pd.DataFrame(curves, columns=['step', 'training_loss', 'cv_loss'])
        fig = cf.plot(x='step', y=['training_loss', 'cv_loss']).get_figure()

        # Save the trained model, so it can be used by the bot
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_directory, "models", args.model_name + ".ckpt")
        print("Training epoch finished, serializing model to {}".format(model_path))
        nn.save(model_path)
        if args.cache == None:
            args.cache = args.model_name + ".ckpt"
        nn._session.close()
        print("Model serialized")


        curve_path = os.path.join(current_directory, "models", args.model_name + "_training_plot.png")
        fig.savefig(curve_path)
    del_sp_data()

def stack_rewards(rewards):
    stacked_rewards = rewards
    return np.asarray([[r] * PLANET_MAX_NUM for r in rewards])

def del_sp_data():
    dir_name = "rlData/"
    dir_list = os.listdir(dir_name)

    for item in dir_list:
        if item.endswith(".data"):
            os.remove(os.path.join(dir_name, item))

def read_data():
    x_data = []
    actions = []
    rewards = []

    for file in os.listdir("rlData/"):
        if file.endswith(".data"):
            with open(os.path.join("rlData/", file), "r") as f:
                f.readline()
                n_planets = int(f.readline())
                x_turn = []
                actions_turn = []
                rewards_turn = []

                done = False
                while not done:
                    faulty = False
                    x_turn.append([])
                    try:
                        actions_turn.append(int(f.readline()[1:-1]))
                        rewards_turn.append(int(f.readline()[1:-1]) * 0.1)
                        for i in range(PLANET_MAX_NUM):
                            line = f.readline()[:-2]
                            if line == "":
                                faulty = True
                                break
                            val_line = line.split(",")
                            x_turn[-1].append([float(val) for val in val_line[:-1]])
                            x_turn[-1][-1].append(float(val_line[-1] == "True"))
                            # Add reward for each ship produced
                            # TODO: Check how changing ownership affects planet production
                            # if len(x_turn >= 2) and x_turn[-1][3] >= 0 and x_turn[-1][3] < x_turn[-2][3]: # If player produces a ship
                            #     rewards_turn[-1] += 0.01
                    except ValueError:
                        faulty = True
                    if faulty:
                        x_turn = x_turn[:-1]
                        rewards_turn = rewards_turn[:-1]
                        actions_turn = actions_turn[:-1]
                        break
                    elif f.readline()[:-1] == "-!":
                        done = True
                        if f.readline() == "WIN":
                            rewards_turn[-1] += 1
                        else:
                            rewards_turn[-1] -= -1
                discount_rewards(rewards_turn)

                x_data.extend(x_turn)
                actions.extend(actions_turn)
                rewards.extend(rewards_turn)

    return np.asarray(x_data), np.asarray(actions), np.asarray(rewards)

def gen_data(samples, cache=None, debug=False):
    # Min of 2 samples
    if samples == 1:
        samples = 2

    for i in range(samples//2):
        print("Generating samples {} and {}...".format(i*2+1, i*2+2))

        command = ["./halite", "-d", "160 160", "-t", "python3 MyBot.py " + str(cache), "python3 MyBotCPU.py " + str(cache)]

        process = Popen(command, stdout=PIPE)
        out, _ = process.communicate()
        out = out.decode("utf-8")

        if debug:
            print(out)

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
                f.write("!\nWIN")
            with open(f2, "a+") as f:
                f.write("!\nLOSS")
        else:
            with open(f1, "a+") as f:
                f.write("!\nLOSS")
            with open(f2, "a+") as f:
                f.write("!\nWIN")

def discount_rewards(rewards):
    running_reward = 0
    for i in reversed(range(len(rewards))):
        running_reward = running_reward * gamma + rewards[i]
        rewards[i] = running_reward

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
