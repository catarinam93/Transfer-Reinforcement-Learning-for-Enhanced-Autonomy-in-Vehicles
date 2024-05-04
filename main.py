from encoder import VariationalEncoder
import argparse
from connection import Connection, logging
from settings import *
from environment import Environment
from ppo import *


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

    env = Environment(client, world, args, encoder)
    env.create_pedestrians()
    env.set_other_vehicles()
    env.get_spawn_ego(EGO_NAME)
    print("Car Spawned")
    env.generate_path()
    print("Path Generated")
    env.get_spawn_sensors(SS_CAMERA)  
    env.get_spawn_sensors(COLLISION_SENSOR) 
    print("Sensors Spawned") 

    # ----------------------------------------------- PPO Algorithm ----------------------------------------------
    ppo = PPO(policy="MlpPolicy", env=env, verbose=1) # Instantiate PPO algorithm
    ppo.learn(total_timesteps=100) # Train the PPO algorithm

    # ppo.save("trained_ppo_model") # Save the trained model
    # ppo.evaluate(env, n_eval_episodes=10) # Evaluate the trained model

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')