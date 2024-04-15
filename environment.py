import random
from connection import carla
import numpy as np
import cv2
import time
from settings import *
import sys
sys.path.append('C:\\Users\\Caty\\Downloads\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner


class Environment():
    def __init__(self, client, world, args):
        self.client = client
        self.world = world
        self.args = args
        self.ego_vehicle = None
        self.start_point = None
        self.spawn_points = None

        self.im_width = 640
        self.im_height = 480

        self.actor_list = []
        self.sensor_list = []
        self.walker_list = []


    # Get the ego vehicle and spawn it in the world on the start point
    def get_spawn_ego(self, ego_name):
        self.spawn_points = self.world.get_map().get_spawn_points() # Get all the spawn points in the map
        ego_bp = self.world.get_blueprint_library().find(ego_name) # Get the blueprint of the ego vehicle
        self.start_point = random.choice(self.spawn_points) # Choose a random spawn point to initiate the ego vehicle
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, self.start_point) # Spawn the ego vehicle in the world


    # Generate the longest route possible regarding the start_point for the ego vehicle to follow
    def generate_path(self):
        point_a = self.start_point.location # Start the path where the car is at

        sampling_resolution = 1 # Resolution of the path
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution) # Create a global route planner instance

        # Pick the longest possible route
        distance = 0
        for loc in self.spawn_points: # For each spawn point in the map
            cur_route = grp.trace_route(point_a, loc.location) # Trace the route from the start point to the spawn point
            if len(cur_route)>distance: # If the route is longer than the previous longest route
                distance = len(cur_route) # Update the distance
                route = cur_route # Update the route

        # Draw the route in the simulation window (Note it does not get into the camera of the car)
        for waypoint in route:
            self.world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=600.0,
                persistent_lines=True)


    # Process the image from the sensor and display it
    def process_img(self, image, sensor_name):
        i = np.array(image.raw_data)  # Convert the image to a numpy array
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # Reshaping the array to the image size
        i3 = i2[:, :, :3]  # Remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
        cv2.imshow(sensor_name, i3)  # Display the image
        cv2.waitKey(1)
        return i3/255.0  # normalize
    

    # Get the spawn points of the sensors and attach them to the ego vehicle
    def get_spawn_sensors(self, rgb_camera, ssc_camera):
        rgb_bp = self.world.get_blueprint_library().find(rgb_camera) # Get the blueprint of the RGB camera
        # change the dimensions of the image
        rgb_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        rgb_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        rgb_bp.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=-5, z=3))

        # spawn the sensor and attach to vehicle.
        rgb_sensor = self.world.spawn_actor(rgb_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensor_list.append(rgb_sensor)
        rgb_sensor.listen(lambda data: self.process_img(data, 'rgb_camera'))
        time.sleep(5)

# ---------------------------------------------------

    # Creating and spawning pedestrians in the world
    def create_pedestrians(self):
        try:
            # Get the available spawn points in the world
            walker_spawn_points = [] 
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform() # Create a spawn point
                loc = self.world.get_random_location_from_navigation() # Get a random location from the navigation
                if (loc != None):
                    spawn_point_.location = loc # Set the location of the spawn point
                    walker_spawn_points.append(spawn_point_) 
            
            # Spawn the walker actor and ai controller and set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.world.get_blueprint_library().filter('walker.pedestrian.*')) # Get the blueprint of the walker
                walker_controller_bp = self.world.get_blueprint_library().find(
                    'controller.ai.walker') # Get the blueprint of the walker controller
                
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false') # Walkers are made visible in the simulation
              
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1])) # Set the speed of the walker (walking, not running)
                else:
                    walker_bp.set_attribute('speed', 0.0) 
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_) # Spawn the walker in the world
             
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker) # Spawn the walker controller in the world
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list) # Get all the actors in the world
            
            self.world.set_pedestrians_cross_factor(0.1) # Set how many pedestrians can cross the road
            
            # Start the motion of the pedestrians
            for i in range(0, len(self.walker_list), 2):
                all_actors[i].start() # Start the walker
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation()) # Set the walker to walk to a random point

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list]) # Destroy the walker actor and controller


# ---------------------------------------------------

# Creating and Spawning other vehciles in the world
    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot mode
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.world.get_map().get_spawn_points()) # Get a random spawn point
                bp_vehicle = random.choice(self.world.get_blueprint_library().filter('vehicle')) # Get the blueprint of the vehicle
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point) # Spawn the vehicle in the world

                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True) # Set the vehicle to autopilot mode
                    self.actor_list.append(other_vehicle)
        
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list]) # Destroy the other vehicles


# ---------------------------------------------------

    # Reset the environment
    def reset(self):
        try: 
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
                        self.sensor_list.clear()
                        self.actor_list.clear()
                        self.walker_list.clear()
                        self.ego_vehicle = None
            
            self.get_spawn_ego(EGO_NAME)
            self.create_pedestrians()
            self.set_other_vehicles()
            self.generate_path()
            self.get_spawn_sensors(RGB_CAMERA, SSC_CAMERA)  

            # Reset all the variables
            self.timesteps = 0 # Reset the timesteps

        except: 
            print("Error in reseting the environment")


# Falta implementar o método de step