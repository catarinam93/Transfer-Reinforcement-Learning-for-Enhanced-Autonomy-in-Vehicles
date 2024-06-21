from encoder import VariationalEncoder
import argparse
from connection import Connection, logging
from settings import *
from environment import Environment
from ppo import *
from stable_baselines3 import PPO
import gymnasium as gym
import os


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


def main():
    args = parse_args()

    # ---------------------------------------------- Generate the Simulation ---------------------------------------------- 
    try:
        client, world = Connection(args.host,args.port, args.town).connect()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    # ----------------------------------------------- Environment and VAE ---------------------------------------------- 
    encoder = VariationalEncoder(LATENT_DIM) 
    encoder.load()

    # Register and create the custom environment
    gym.register(id='Carla_Env', entry_point=lambda: Environment(client, world, args, encoder))
    env = gym.make('Carla_Env')

    # ----------------------------------------------- PPO Algorithm ----------------------------------------------
    # Define directories for saving models and logging
    models_dir = "models/PPO"
    logdir = "tensorboard"

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Define training parameters
    TIMESTEPS = 50000
    iters = 0
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    print("Model Created")
    for i in range(iters, 5):
        print("================================ Iteration", i, " ================================")
        # Check if there are previously trained models
        model_path = f"{models_dir}/{i}.zip"
        if os.path.exists(model_path):
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