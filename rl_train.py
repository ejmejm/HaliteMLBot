import argparse

import numpy as np
import pandas as pd
from reinforcementLearner.parsing import parse
from subprocess import Popen, PIPE

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

    ep_history = []

    run_command(["./halite", "-d", "160 160", "-t", "python3 MyBot.py", "python3 MyBotCPU.py"])

    # process = Popen(["../halite", "-d", "160 160", "-t", "python3 MyBot.py", "python3 MyBotCPU.py"], stdout=PIPE, stderr=PIPE)
    #
    # output, err = process.communicate()
    #
    # print("OUTPUT: {}".format(output))
    # print()
    # print("ERROR: {}".format(err))
    # print()
    # print("Finished")

# https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
def run_command(command):
    process = Popen(command, stdout=PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

def playthrough(neural_net):

    # Initialize the game.
    game = hlt.Game(self._name)

    while True:
        # Update the game map.
        game_map = game.update_map()
        print(type(game_map))
        start_time = time.time()

        # Produce features for each planet.
        features = self.produce_features(game_map)

        # Find predictions which planets we should send ships to.
        predictions = neural_net.predict(features)

        # Use simple greedy algorithm to assign closest ships to each planet according to predictions.
        ships_to_planets_assignment = self.produce_ships_to_planets_assignment(game_map, predictions)

        # Produce halite instruction for each ship.
        instructions = self.produce_instructions(game_map, ships_to_planets_assignment, start_time)

        # Send the command.
        game.send_command_queue(instructions)

if __name__ == "__main__":
    main()
