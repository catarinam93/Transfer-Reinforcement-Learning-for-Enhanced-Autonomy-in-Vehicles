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

sys.path.append('C:\\Users\\Caty\\Downloads\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner


class Environment(gym.Env):
    def __init__(self, client, world, args, encoder):
        self.client = client
        self.world = world # Get the Town of the simulation
        self.args = args
        self.ego_vehicle = None
        self.start_point = None # Start point of the ego vehicle
        self.spawn_points = None # Spawn points of the ego vehicle
        self.route = None # Route of the ego vehicle
        self.collision_occured = 0 # Collision flag

        self.encoder = encoder # Encoder of the environment
        self.n_camera_features = 97 # Number of camera features
        self.features_accumulator = [] # Variable to store the camera features

        self.actor_list = []
        self.sensor_list = []
        self.walker_list = []
        self.camera = None
        self.collision_sensor = None

        # Limits for the action space
        self.linear_velocity_low = 0.0
        self.linear_velocity_high = 1.0
        self.break_low = 0.0
        self.break_high = 1.0
        self.angular_velocity_low = -1.0
        self.angular_velocity_high = 1.0

        # Limits for the observation space
        self.camera_features_low = -np.inf
        self.camera_features_high = np.inf
        self.distance_low = 0
        self.distance_high = np.inf
        self.angle_low = -np.pi
        self.angle_high = np.pi
        self.len_route_low = 0
        self.len_route_high = np.inf
        self.collision_occured_low = 0
        self.collision_occured_high = 1

        # Size of the observation space
        self.camera_features_size = 95
        self.distance_size = 1
        self.angle_size = 1
        self.len_route_size = 1
        self.collision_occured_size = 1

        # Total size of the observation space
        self.total_observation_size = self.camera_features_size + self.distance_size + self.angle_size + self.len_route_size + self.collision_occured_size

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


# ---------------------- Sensors -----------------------------
    
    def on_collision(self):
        print("Collision detected")
        self.collision_occured = 1
    
    # Process the image from the sensor and display it
    def process_img(self, image):
        try:
            i = np.array(image.raw_data)  # Convert the image to a numpy array
            i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # Reshaping the array to the image size
            i3 = i2[:, :, :3]  # Remove the alpha channel
            normalized_image = i3 / 255.0  # Normalize the image
            cv2.imshow("SS_CAM", i3)  # Display the image
            cv2.waitKey(1)

            image_tensor = torch.tensor(normalized_image, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2)  # Convert the image to a tensor

            # Check if CUDA is available and move encoder and tensor to the appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.encoder.to(device)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                features = self.encoder(image_tensor)  # Get the latent features of the image

            self.features_accumulator.append(features)  # Append the features to the accumulator

        except Exception as e:
            print(f"Error in process_img: {e}")
        return None


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

        else: # If the sensor is a collision sensor
            spawn_point = carla.Transform(carla.Location(x=0, z=1))

        # Spawn the sensor, attaching them to vehicle.
        sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensor_list.append(sensor)
        if sensor_name == SS_CAMERA:
            self.camera = sensor
        else:
            self.collision_sensor = sensor

# ------------------------ Observations ---------------------------
    # Get the observations of the environment
    def get_obs(self):
        distance, angle = self.distance_angle_towards_waypoint()
        self.camera.listen(lambda data: self.process_img(data))
        self.collision_sensor.listen(lambda event: self.on_collision())
        

        while len(self.features_accumulator) == 0:
            time.sleep(0.05)

        if self.features_accumulator:
            average_features = sum(self.features_accumulator) / len(self.features_accumulator) # Get the average of the features
            self.features_accumulator = [] # Reset the accumulator
        # print("camera features: ", average_features)
        # print("distance: ", distance)
        # print("angle: ", angle)
        # print("len route: ", len(self.route))
        # print("collision: ", self.collision_occured)
        # Concatenate and flatten all observation components into a single array
        observation = np.concatenate([average_features.cpu().numpy().flatten(), np.array([distance, angle, len(self.route), self.collision_occured])])
        
        return observation


    # Get the distance and angle towards the next waypoint
    def distance_angle_towards_waypoint(self):
        if not self.route:
            return None, None  # Return None if there is no route

        current_position = np.array([self.ego_vehicle.get_location().x, self.ego_vehicle.get_location().y, self.ego_vehicle.get_location().z])
        next_waypoint_position = np.array([self.route[0][0].transform.location.x, self.route[0][0].transform.location.y, self.route[0][0].transform.location.z])
        distance = np.linalg.norm(next_waypoint_position - current_position)  # Calculate the distance between the vehicle and the next waypoint

        # Verify if the vehicle has passed the waypoint
        if distance < 0:
            self.route = self.route[1:]  # Remove the waypoint from the route
            return self.distance_angle_towards_waypoint()  # Recalculate the distance and angle towards the next waypoint
        else:
            # Get the rotation of the vehicle and the next waypoint
            current_rotation = np.array([self.ego_vehicle.get_transform().rotation.pitch , self.ego_vehicle.get_transform().rotation.yaw , self.ego_vehicle.get_transform().rotation.roll])
            next_waypoint_rotation = np.array([self.route[0][0].transform.rotation.pitch , self.route[0][0].transform.rotation.yaw , self.route[0][0].transform.rotation.roll])

            # Compute the norms of the vectors
            current_rotation_norm = np.linalg.norm(current_rotation)
            next_waypoint_rotation_norm = np.linalg.norm(next_waypoint_rotation)

            # Add a check for zero or near-zero norm values to avoid division by zero
            if current_rotation_norm > 0 and next_waypoint_rotation_norm > 0:
                current_rotation_normalized = current_rotation / current_rotation_norm
                next_waypoint_rotation_normalized = next_waypoint_rotation / next_waypoint_rotation_norm
            else:
                # Handle the case when the norm is zero or near-zero
                current_rotation_normalized = np.zeros_like(current_rotation)
                next_waypoint_rotation_normalized = np.zeros_like(next_waypoint_rotation)

            dot_product = np.dot(current_rotation_normalized, next_waypoint_rotation_normalized) # Calculate the dot product between the vehicle and the next waypoint

            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0)) # Calculate the angle between the vehicle and the next waypoint
            
            return distance, angle_rad



# ---------------------------------------------------
    # Reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Reset the environment

        if len(self.actor_list) != 0 or len(self.sensor_list) != 0: # Destroy the existing environment
                    self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                    self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                    self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
                    self.sensor_list.clear()
                    self.actor_list.clear()
                    self.walker_list.clear()
                    self.ego_vehicle = None
        
        # Create a new environment
        self.get_spawn_ego(EGO_NAME)
        self.create_pedestrians()
        self.set_other_vehicles()
        self.generate_path()
        self.get_spawn_sensors(SS_CAMERA)  
        # self.get_spawn_sensors(COLLISION_SENSOR)  

        # Reset all variables
        self.timesteps = 0 # Reset the timesteps
        self.reward = 0 # Reset the reward
        # self.collision_occured = False # Reset the collision flag
        self.terminated = False # Reset the termination flag
        observation = self.get_obs()
        print("Reset observation: ", len(observation))
        return observation, {}


# ----------------------  Step and Reward -----------------------------
    def calculate_reward(self, linear_velocity, distance_to_next_waypoint, angle_toward_next_waypoint, len_route, collision_occured):
        if collision_occured:
            self.reward += COLLISION_PENALTY
            self.terminated = True
            return
        
        elif len_route == 0:
            self.reward += DESTINATION_REWARD
            self.terminated = True
            return
        
        elif angle_toward_next_waypoint > THETA:
            self.reward += ANGLE_PENALTY
        
        elif linear_velocity > MAX_SPEED or linear_velocity == 0:
            self.reward +=  SPEED_PENALTY

        elif distance_to_next_waypoint > WAYPOINT_THRESHOLD:
            self.reward += WAYPOINT_REWARD
        
        else: self.reward += NEUTRAL_REWARD

    def step(self, action):
        linear_velocity = action[0]  
        angular_velocity = action[1] 
        break_value = action[2]
        # Apply the control to the ego vehicle
        ego_vehicle_control = carla.VehicleControl(throttle=float(linear_velocity), steer = float(angular_velocity), brake = float(break_value))
        self.ego_vehicle.apply_control(ego_vehicle_control)

        # Get the observations
        observation = self.get_obs()
        distance_to_next_waypoint = observation[0]
        angle_toward_next_waypoint = observation[1] 
        len_route = observation[2]
        collision_occured = observation[3]

        self.calculate_reward(linear_velocity, distance_to_next_waypoint, angle_toward_next_waypoint, len_route, collision_occured)

        return observation, self.reward, self.terminated



# -------------------- Setting the rest of the environment -------------------------------
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