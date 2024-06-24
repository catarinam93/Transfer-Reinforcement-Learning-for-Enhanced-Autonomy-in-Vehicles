"""
This script defines an environment simulation using the CARLA library and the Gymnasium interface. 

The environment is designed to train and evaluate an autonomous vehicle (ego vehicle) in an urban scenario. It includes functionality 
for spawning and controlling the ego vehicle, generating a route for it to follow, and setting up sensors to provide observations. 

Additionally, it manages the spawning of pedestrians and other vehicles to create a dynamic and realistic simulation. 

The environment is equipped to handle collisions, compute rewards, and reset itself for iterative training cycles.
"""

import random
from connection import carla
import numpy as np
import cv2
import time
import gymnasium as gym
from gymnasium import spaces
from settings import *
import sys
import torch
from collections import deque

sys.path.append('C:\\Users\\Caty\\Downloads\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner


# Environment class
class Environment(gym.Env):
    def __init__(self, client, world, args, encoder):
        self.client = client
        self.world = world # Get the Town of the simulation
        self.args = args # Get the arguments of the simulation
        self.ego_vehicle = None # Ego vehicle
        self.start_point = None # Start point of the ego vehicle
        self.spawn_points = None # Spawn points of the ego vehicle
        self.route = None # Route of the ego vehicle
        self.initial_len_route = 0 # Initial length of the route
        self.collision_occured = 0 # Collision flag
        self.number_of_collisions = 0 # Number of collisions
        self.waypoints_route_completed = 0 # Percentage of the route completed

        self.encoder = encoder # Encoder of the environment
        self.n_camera_features = 97 # Number of camera features
        self.features_accumulator = deque(maxlen=5) # Buffer to store the camera features

        self.actor_list = [] # List of actors in the environment
        self.sensor_list = [] # List of sensors in the environment
        self.walker_list = [] # List of walkers in the environment
        self.camera = None # Camera sensor
        self.collision_sensor = None # Collision sensor
        self.bev_camera = None # Bird's eye view camera sensor

        # Limits for the action space
        self.linear_velocity_low = 0.0 # Low limit of the linear velocity
        self.linear_velocity_high = 1.0 # High limit of the linear velocity
        self.break_low = 0.0 # Low limit of the break
        self.break_high = 1.0 # High limit of the break
        self.angular_velocity_low = -1.0 # Low limit of the angular velocity
        self.angular_velocity_high = 1.0 # High limit of the angular velocity

        # Limits for the observation space
        self.camera_features_low = -np.inf # Low limit of the camera features
        self.camera_features_high = np.inf # High limit of the camera features
        self.distance_low = 0 # Low limit of the distance
        self.distance_high = np.inf # High limit of the distance
        self.angle_low = -np.pi # Low limit of the angle
        self.angle_high = np.pi # High limit of the angle
        self.len_route_low = 0 # Low limit of the length of the route
        self.len_route_high = np.inf # High limit of the length of the route
        self.collision_occured_low = 0 # Low limit of the collision flag
        self.collision_occured_high =  1 # High limit of the collision flag

        # Size of the observation space
        self.camera_features_size = 95 # Size of the camera features
        self.distance_size = 1 # Size of the distance
        self.angle_size = 1 # Size of the angle
        self.len_route_size = 1 # Size of the length of the route
        self.collision_occured_size = 1 # Size of the collision flag

        # Total size of the observation space
        self.total_observation_size = self.camera_features_size + self.distance_size + self.angle_size + self.len_route_size + self.collision_occured_size

        # Action and observation space
        self.action_space = spaces.Box(low=np.array([self.linear_velocity_low, self.angular_velocity_low, self.break_low]), 
                                        high=np.array([self.linear_velocity_high, self.angular_velocity_high, self.break_high]), 
                                        dtype=np.float64)

        self.observation_space = spaces.Box(low=np.array([self.camera_features_low] * self.camera_features_size + [self.distance_low] + [self.angle_low] + [self.len_route_low] + [self.collision_occured_low]), 
                                            high=np.array([self.camera_features_high] * self.camera_features_size + [self.distance_high] + [self.angle_high] + [self.len_route_high] + [self.collision_occured_high]), 
                                            dtype=np.float64)


# ---------------------- Ego Vehicle -----------------------------
    # Get the ego vehicle and spawn it in the world on the start point
    def get_spawn_ego(self, ego_name):
        self.spawn_points = self.world.get_map().get_spawn_points() # Get all the spawn points in the map
        ego_bp = self.world.get_blueprint_library().find(ego_name) # Get the blueprint of the ego vehicle
        self.start_point = random.choice(self.spawn_points) # Choose a random spawn point to initiate the ego vehicle to avoid overfitting
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, self.start_point) # Spawn the ego vehicle in the world


# ---------------------- Route Generation -----------------------------
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
                self.route = cur_route # Update the route

        # Draw the route in the simulation window (Note: it does not get into the camera of the car)
        for waypoint in self.route:
            self.world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=600.0,
                persistent_lines=True)
            
        self.initial_len_route = len(self.route) # Get the initial length of the route for percentage calculation


# ---------------------- Sensors -----------------------------
    # Function to handle the collision sensor
    def on_collision(self):
        self.collision_occured = 1
    
    # Process the image from the SSC and Collision sensors and display it
    def process_img(self, image):
        try:
            i = np.array(image.raw_data)  # Convert the image to a numpy array
            i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # Reshaping the array to the image size
            i3 = i2[:, :, :3]  # Remove the alpha channel
            normalized_image = i3 / 255.0  # Normalize the image

            window_name = "SS_CAM"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create a window
        
            scale_factor = 3  # Scale factor for the window
            new_width = IM_WIDTH * scale_factor
            new_height = IM_HEIGHT * scale_factor
            cv2.resizeWindow(window_name, new_width, new_height) # Resize the window

            cv2.imshow(window_name, i3)  # Display the image
            cv2.waitKey(1) # Wait for a key press

            image_tensor = torch.tensor(normalized_image, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2)  # Convert the image to a tensor

            # Check if CUDA is available and move encoder and tensor to the appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if CUDA is available
            self.encoder.to(device) # Move the encoder to the device
            image_tensor = image_tensor.to(device) # Move the tensor to the device

            with torch.no_grad():
                features = self.encoder(image_tensor)  # Get the latent features of the image

            self.features_accumulator.append(features)  # Append the features to the accumulator

        except Exception as e:
            print(f"Error in process_img: {e}")
        return None

    # Process the image from the RGB camera and display it
    def display_img(self, image):
        try:
            i = np.array(image.raw_data).reshape((image.height, image.width, 4)) # Convert the image to a numpy array
            i = i[:, :, :3] # Remove the alpha channel

            window_name = "Bird View Camera"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create a window
            
            cv2.imshow(window_name, i) # Display the image
            cv2.waitKey(1) # Wait for a key press

        except Exception as e:
            print(f"Error in display_img: {e}")

    # Get the spawn points of the sensors and attach them to the ego vehicle
    def get_spawn_sensors(self, sensor_name):
        sensor_bp = self.world.get_blueprint_library().find(sensor_name) # Get the blueprint of the SS camera
        
        if sensor_name == SS_CAMERA:
            # Change the dimensions of the image
            sensor_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
            sensor_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
            sensor_bp.set_attribute('fov', '110')

            # Adjust sensor relative to vehicle
            spawn_point = carla.Transform(carla.Location(x=-5, z=3))

        elif sensor_name == RGB_CAMERA:
            # Bird's eye view camera settings
            sensor_bp.set_attribute('image_size_x', '320')
            sensor_bp.set_attribute('image_size_y', '320')
            sensor_bp.set_attribute('fov', '110')

            # Adjust sensor relative to vehicle
            spawn_point = carla.Transform(carla.Location(x=0, z=10), carla.Rotation(pitch=-90))

        else: # If the sensor is a collision sensor
            spawn_point = carla.Transform(carla.Location(x=0, z=1))

        # Spawn the sensor, attaching them to vehicle.
        sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensor_list.append(sensor) # Append the sensor to the sensor list

        # Set up the listener for the sensor
        if sensor_name == SS_CAMERA:
            self.camera = sensor
            self.camera.listen(lambda data: self.process_img(data)) # Listen to the sensor data
        elif sensor_name == RGB_CAMERA:
            self.bev_camera = sensor
            self.bev_camera.listen(lambda data: self.display_img(data)) # Listen to the sensor data
        else:
            self.collision_sensor = sensor
            self.collision_sensor.listen(lambda event: self.on_collision()) # Listen to the sensor data

# ------------------------ Observations ---------------------------
    # Get the observations of the environment
    def get_obs(self):
        distance, angle = self.distance_angle_towards_waypoint() # Get the distance and angle towards the next waypoint
        
        # Wait for the features to be accumulated
        while len(self.features_accumulator) == 0:
            time.sleep(0.05)

        # Get the average of the features for the last 5 frames for several reasons:
        # 1. To reduce the noise and variations in the features
        # 2. To capture the temporal information in the features (5 frames)
        # 3. To provide a more stable observation (reducing temporal fluctuations)
        # 4. To reduce the dimensionality of the observation
        # 5. To provide a more informative observation
        # 6. To provide a more robust observation
        if self.features_accumulator:
            average_features = sum(self.features_accumulator) / len(self.features_accumulator)

        # Concatenate and flatten all observation components into a single array
        observation = np.concatenate([average_features.cpu().numpy().flatten(), np.array([distance, angle, len(self.route), self.collision_occured])])
        
        return observation


    # Get the distance and angle towards the next waypoint
    def distance_angle_towards_waypoint(self):  
        current_position = np.array([self.ego_vehicle.get_location().x, self.ego_vehicle.get_location().y, self.ego_vehicle.get_location().z]) # Get the current position of the vehicle
        next_waypoint_position = np.array([self.route[0][0].transform.location.x, self.route[0][0].transform.location.y, self.route[0][0].transform.location.z]) # Get the position of the next waypoint
        distance = np.linalg.norm(next_waypoint_position - current_position)  # Calculate the distance between the vehicle and the next waypoint

        # Verify if the vehicle has passed the waypoint
        # If the distance is negative, the vehicle has passed the waypoint
        if distance < 0:
            self.route = self.route[1:]  # Remove the waypoint from the route
            self.waypoints_route_completed += 1  # Increment the percentage of the route completed
            return self.distance_angle_towards_waypoint()  # Recalculate the distance and angle towards the next waypoint

        else:
            # Get the rotation of the vehicle and the next waypoint
            current_rotation = np.array([self.ego_vehicle.get_transform().rotation.pitch , self.ego_vehicle.get_transform().rotation.yaw , self.ego_vehicle.get_transform().rotation.roll]) # Get the rotation of the vehicle
            next_waypoint_rotation = np.array([self.route[0][0].transform.rotation.pitch , self.route[0][0].transform.rotation.yaw , self.route[0][0].transform.rotation.roll]) # Get the rotation of the next waypoint

            # Compute the norms of the vectors
            current_rotation_norm = np.linalg.norm(current_rotation)
            next_waypoint_rotation_norm = np.linalg.norm(next_waypoint_rotation)

            # Add a check for zero or near-zero norm values to avoid division by zero
            if current_rotation_norm > 0 and next_waypoint_rotation_norm > 0:
                current_rotation_normalized = current_rotation / current_rotation_norm # Normalize the rotation of the vehicle
                next_waypoint_rotation_normalized = next_waypoint_rotation / next_waypoint_rotation_norm # Normalize the rotation of the next waypoint
            
            else:
                # Handle the case when the norm is zero or near-zero
                current_rotation_normalized = np.zeros_like(current_rotation) 
                next_waypoint_rotation_normalized = np.zeros_like(next_waypoint_rotation)

            dot_product = np.dot(current_rotation_normalized, next_waypoint_rotation_normalized) # Calculate the dot product between the vehicle and the next waypoint

            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0)) # Calculate the angle between the vehicle and the next waypoint
            
            return distance, angle_rad


# ---------------------- Reset Environment -----------------------------
    # Reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Reset the environment

        # Destroy the existing environment
        try:
            if self.sensor_list:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])

            if self.actor_list:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

            if self.walker_list:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in range(1, len(self.walker_list), 2)])
            
            time.sleep(0.5) # Wait for the environment to be destroyed

            # Clear the lists
            self.sensor_list.clear()
            self.actor_list.clear()
            self.walker_list.clear()
            self.ego_vehicle = None
            self.camera = None
            self.collision_sensor = None

        except Exception as e:
            print(f"Error during environment reset: {e}")
            # Verify destruction
            all_actors = self.world.get_actors()
            remaining_sensors = [actor.id for actor in all_actors if actor.id in self.sensor_list]
            remaining_actors = [actor.id for actor in all_actors if actor.id in self.actor_list]
            remaining_walkers = [actor.id for actor in all_actors if actor.id in self.walker_list]
           
            if not remaining_sensors and not remaining_actors and not remaining_walkers:
                print("All was destroyed successfully.")
            else: 
                print("Some actors were not destroyed.")

        # Create a new environment
        self.get_spawn_ego(EGO_NAME)
        self.create_pedestrians()
        self.set_other_vehicles()
        self.generate_path()
        self.get_spawn_sensors(SS_CAMERA)  
        self.get_spawn_sensors(COLLISION_SENSOR)
        self.get_spawn_sensors(RGB_CAMERA)

        # Reset all variables
        self.timesteps = 0 
        self.reward = 0 
        self.collision_occured = 0 
        self.terminated = False
        self.waypoints_route_completed = 0
        self.features_accumulator.clear() 
        self.number_of_collisions = 0 
        observation = self.get_obs()

        return observation, {}
    

# ----------------------  Step and Reward -----------------------------
    # Calculate the reward based on the environment state
    def calculate_reward(self, linear_velocity, distance_to_next_waypoint, angle_toward_next_waypoint, len_route, collision_occured):
        # If there was a collision, terminate the episode and penalize the reward
        if collision_occured:
            self.number_of_collisions += 1
            self.reward += COLLISION_PENALTY
            self.terminated = True
            return
        
        # If the vehicle reached the destination, reward the agent
        elif len_route == 0:
            self.reward += DESTINATION_REWARD
            self.terminated = True
            return
        
        # If the angle between the vehicle and the next waypoint is to big, penalize the agent
        elif angle_toward_next_waypoint > THETA:
            self.reward += ANGLE_PENALTY
        
        # If the vehicle is moving to fast (above max speed) or is not moving, penalize the agent
        # MAX_SPEED = 13.89 -> 50 km/h in m/s (approximate value)
        elif linear_velocity > MAX_SPEED or linear_velocity == 0:
            self.reward +=  SPEED_PENALTY

        # If the vehicle has reached the next waypoint, reward the agent
        # WAYPOINT_THRESHOLD = 0.1 -> Proximity distance to consider waypoint reached (approximate value)
        elif distance_to_next_waypoint > WAYPOINT_THRESHOLD:
            self.reward += WAYPOINT_REWARD
        
        # If the vehicle is moving towards the next waypoint, reward the agent
        else: self.reward += NEUTRAL_REWARD

    # Step function of the environment
    def step(self, action):
        # Get the action components
        linear_velocity = action[0]  
        angular_velocity = action[1] 
        break_value = action[2]
        
        # Apply the control to the ego vehicle
        ego_vehicle_control = carla.VehicleControl(throttle=float(linear_velocity), steer = float(angular_velocity), brake = float(break_value))
        self.ego_vehicle.apply_control(ego_vehicle_control)
        
        # Wait for some time to allow the vehicle to move
        time.sleep(0.5)
        
        # Get the observations
        observation = self.get_obs()
        distance_to_next_waypoint = observation[self.camera_features_size]
        angle_toward_next_waypoint = observation[self.camera_features_size + 1]
        len_route = observation[self.camera_features_size + 2]
        collision_occured = observation[self.camera_features_size + 3]

        self.calculate_reward(linear_velocity, distance_to_next_waypoint, angle_toward_next_waypoint, len_route, collision_occured) # Calculate the reward

        return observation, self.reward, self.terminated, False, {}  # Return the observation, reward, terminated flag, truncated flag, and info dictionary


# -------------------- Setting the rest of the environment -------------------------------
    # Creating and spawning pedestrians in the world
    def create_pedestrians(self):
        try:
            # Get the available spawn points in the world
            walker_spawn_points = [] 
            for i in range(NUMBER_OF_PEDESTRIAN + 1):
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


    # Creating and Spawning other vehicles in the world
    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot mode
            for _ in range(0, NUMBER_OF_VEHICLES+1):
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