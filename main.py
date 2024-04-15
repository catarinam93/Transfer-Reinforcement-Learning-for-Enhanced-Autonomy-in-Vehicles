# from carla import VehicleLightState as vls
# from numpy import random
# from stable_baselines3 import PPO
import argparse
from connection import Connection, logging, carla
from settings import EGO_NAME, RGB_CAMERA, SSC_CAMERA
from environment import Environment


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


def main():
    args = parse_args()

    # ---------------------------------------------- Generate the Simulation ---------------------------------------------- 
    try:
        client, world = Connection(args.host,args.port, args.town).connect()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    # ----------------------------------------------- Spawn and Create Route ---------------------------------------------- 
    env = Environment(client, world, args)
    env.create_pedestrians()
    env.set_other_vehicles()
    env.get_spawn_ego(EGO_NAME)
    env.generate_path()
    env.get_spawn_sensors(RGB_CAMERA, SSC_CAMERA)   
    # env.create_pedestrians()
    # spawn_points = world.get_map().get_spawn_points()
    # spawn_point = spawn_points[247] # A spawn point at the entrance of the roundabout - see image on the report
    # ''' Sugestion of ending points
    # - 221 -> first exit
    # - 85 -> second exit
    # - 212 -> third exit
    # - 250 -> fourth exit
    # '''
    # # Draw the spawn point locations as numbers in the map with a life time of 10 seconds
    # for i, spawn_point in enumerate(spawn_points):
    #     world.debug.draw_string(spawn_point.location, str(i), life_time=10)
    #     print("i: ", i, "spawn_point: ", spawn_point)
    
    
    # # ---------------------------------------------- Vehicle ---------------------------------------------- 
    
    # # ego_bp = random.choice(world.get_blueprint_library().filter('*vehicle*')) # If I wanted a random vehicle 
    # ego_bp = world.get_blueprint_library().find('vehicle.mini.cooper_s_2021')
    # ego_bp.set_attribute('role_name', 'hero')
    # ego_vehicle = world.spawn_actor(ego_bp, spawn_point) # The veichle is spawn at the spawn 


    # # ---------------------------------------------- Sensors ---------------------------------------------- 
    # ss_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation') # A camera with Semantic Segmentation which gives us 12 different classifications, e.g., lane marker, sidewalk, road, pedestrian, etc.
    # co_bp = world.get_blueprint_library().find('sensor.other.collision') # A collision sensor
    # li_bp = world.get_blueprint_library().find('sensor.other.lane_invasion') # A lane invasion sensor

    # transform = carla.Transform(carla.Location(x=0.8, z=1.7)) # How do I know the proper values to insert here?

    # # Spawn the sensors on the ego vehicle
    # ss_sensor = world.spawn_actor(ss_bp, transform, attach_to=ego_vehicle)
    # co_sensor = world.spawn_actor(co_bp, transform, attach_to=ego_vehicle)
    # li = world.spawn_actor(li_bp, transform, attach_to=ego_vehicle)


    # # ---------------------------------------------- Destruction of the vehicle ---------------------------------------------- 
    # # When a final spawn point is reached - destroy the vehicle
    # #     print('\ndestroying the vehicle')
    # #     client.apply_batch([carla.command.DestroyActor(ego_vehicle)])
    # #     time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

