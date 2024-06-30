'''
All the parameters used in the Simulation has been documented here.

Easily modifiable paramters with the quick access in this settings.py file 
to achieve quick modifications especially during the training sessions.

Names of the parameters are self-explanatory therefore elimating the use of further comments.
'''

import numpy as np

ENV_MAX_STEPS = 1000 # Maximum number of steps per episode

EGO_NAME = 'vehicle.mini.cooper_s_2021'

NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10

LATENT_DIM = 95 
IM_WIDTH = 160
IM_HEIGHT = 80

SS_CAMERA = 'sensor.camera.semantic_segmentation'
COLLISION_SENSOR = 'sensor.other.collision'
LANE_SENSOR = 'sensor.other.lane_invasion'

COLLISION_PENALTY = -100.0  # High penalty for collisions
DESTINATION_REWARD = 10000.0  # High reward upon reaching the destination
ANGLE_PENALTY = -40.0  # Moderate penalty for large angles relative to the next waypoint
SPEED_PENALTY = -10.0  # Moderate penalty for very high
NOT_MOVE_SPEED = -50.0  # Moderate penalty for not moving
WAYPOINT_REWARD = 100.0  # Moderate reward for approaching a waypoint
NEUTRAL_REWARD = 30.0  # Neutral reward for other states
LANE_INVASION_PENALTY = - 40.0  # Moderate penalty for lane invasions

THETA = np.pi / 4  # Angle threshold for angle penalty (45 degrees)
MAX_SPEED = 13.89  # 50 km/h in m/s (approximate value)
# The average distance between waypoints is 0.9792900460064606, therefore the threshold is set to the average distance * 0.1 ~= 0.1
WAYPOINT_THRESHOLD = 0.1  # Proximity distance to consider waypoint reached (approximate value)
