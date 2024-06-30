'''This script tests the performance of PPO trained model in a new unseen environment.
It loads the models, runs them in the environment, and generates the graphs for the performance metrics.'''

from encoder import VariationalEncoder
import argparse
from connection import Connection, logging
from settings import *
from environment import Environment
from stable_baselines3 import PPO
import gymnasium as gym
import os
import matplotlib.pyplot as plt
import time

# Function to parse the arguments
def parse_args():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--town',
        type=str,
        default="Town10HD",
        help='Town of the simulation (default: Town10HD)')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')

    args = argparser.parse_args()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    return args

def test_main():
    args = parse_args()

    # ---------------------------------------------- Generate the Simulation ----------------------------------------------
    try:
        client, world = Connection(args.host, args.port, args.town).connect()
        logging.info("Connection has been set up successfully.")
    except Exception as e:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    # ----------------------------------------------- Environment and VAE ----------------------------------------------
    encoder = VariationalEncoder(LATENT_DIM)
    encoder.load()

    # Define directories for saving the metrics's graphs
    graphs_dir = "graphs"

    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    # Register and create the custom environment
    gym.register(id='Carla_Env', entry_point=lambda: Environment(client, world, args, encoder, graphs_dir), max_episode_steps=ENV_MAX_STEPS)
    env = gym.make('Carla_Env')

    # Load the model
    model = PPO.load("models/Town02/PPO/No_Traffic/5", env=env)

    # Run the environment with the loaded model
    episodes = 10
    total_reward = 0
    total_timesteps = 0

    start_time = time.time()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info, _ = env.step(action)
            episode_reward += rewards
            total_timesteps += 1
        total_reward += episode_reward
        print(f"Episode {ep+1} reward: {episode_reward}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate average reward
    avg_reward = total_reward / episodes

    # Print the results
    print(f"Model: {args.model_path}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Average reward per episode: {avg_reward}")

if __name__ == '__main__':
    try:
        test_main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
