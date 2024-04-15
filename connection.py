import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import logging

class Connection():
    def __init__(self, host, port, town):
        self.host = host
        self.port = port
        self.town = town
        self.client = None
        self.world = None

    def connect(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            print(self.town)
            # print(self.client.get_available_maps())
            self.world = self.client.load_world(self.town) 
            return self.client, self.world
        
        except Exception as e:
            logging.error("An error occurred while trying to connect to the server: \n" + str(e))
