from settings import *
import carla
import numpy as np
import cv2

class Camera():
    def __init__(self, vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        self.sensor.listen(lambda data: self.process_img(data, 'rgb_camera'))


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

    # Process the image from the sensor and display it
    def process_img(self, image, sensor_name):
        i = np.array(image.raw_data)  # Convert the image to a numpy array
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # Reshaping the array to the image size
        i3 = i2[:, :, :3]  # Remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
        cv2.imshow(sensor_name, i3)  # Display the image
        cv2.waitKey(1)
        return i3/255.0  # normalize
    

