'''This script implements a reinforcement learning framework for training an autonomous vehicle using Proximal Policy Optimization (PPO) 
within a simulated CARLA environment. 

It begins by parsing command-line arguments to configure the simulation, such as the server's IP address, port, simulation town, seed 
for the pedestrians module, and whether to use asynchronous mode. 

The script establishes a connection to the CARLA server, initializes a variational encoder for state representation, and registers a 
custom environment with Gymnasium.

The main function then sets up directories for saving models and logs, defines training parameters (including the number of timesteps and
iterations), and creates the PPO model. It iteratively trains the model, saving it after each iteration, and collects data on the number 
of collisions and route completion percentage for visualization. This data is plotted and saved as graphs at the end of each training 
iteration, providing a visual representation of the model's performance over time. The script is designed to handle interruptions 
gracefully and includes logging for monitoring the connection and training progress.
'''
from encoder import VariationalEncoder
import argparse
from connection import Connection, logging
from settings import *
from environment import Environment
from stable_baselines3 import PPO
import gymnasium as gym
import os

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
        default="Town01", 
        help='Town of the simulation (default: Town01)')
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

# Main function
def main():
    args = parse_args()

    # ---------------------------------------------- Generate the Simulation ---------------------------------------------- 
    try:
        client, world = Connection(args.host,args.port, args.town).connect()
        logging.info("Connection has been setup successfully.")
    except Exception as e:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    # ----------------------------------------------- Environment and VAE ---------------------------------------------- 
    encoder = VariationalEncoder(LATENT_DIM)
    encoder.load()

    # Define directories for saving models and logging
    models_dir = "models/Town01/PPO"
    graphs_dir = "graphs"

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    # Register and create the custom environment
    gym.register(id='Carla_Env', entry_point=lambda: Environment(client, world, args, encoder, graphs_dir), max_episode_steps=ENV_MAX_STEPS)
    env = gym.make('Carla_Env')

    # ----------------------------------------------- PPO Algorithm ----------------------------------------------

    # Define training parameters
    TIMESTEPS = 10000  # 10,000 timesteps per iteration
    iters = 100  # Total of 100 iterations

    for i in range(iters):
        print("================================ Iteration", i + 1, " ================================")
        
        # Check if there are previously trained models
        model_path = f"{models_dir}/{i}.zip"
        if i == 0 or not os.path.exists(model_path):
            # On the first iteration or if no model is found, create a new model
            model = PPO("MlpPolicy", env, verbose=1)
        else:
            # Load the trained model from the previous iteration
            print(f"Loading the trained model from {model_path}")
            model = PPO.load(model_path, env)
        
        # Train the model
        print("Training the model")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="ppo")

        # Save the model after each iteration
        print(f"Saving the model to {models_dir}/{(i + 1)}")
        model.save(f"{models_dir}/{(i + 1)}")
   
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
