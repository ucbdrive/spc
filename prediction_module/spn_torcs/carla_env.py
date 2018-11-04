from __future__ import print_function, division

import random
import numpy as np
import cv2

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.image_converter import labels_to_array


def default_settings():
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=1,  # random.choice([1, 3, 7, 8, 14]),
        PlayerVehicle='/Game/Blueprints/Vehicles/Mustang/Mustang.Mustang_C',
        QualityLevel='Epic')
    settings.randomize_seeds()

    camera_RGB = Camera('CameraRGB')
    camera_RGB.set_image_size(256, 256)
    camera_RGB.set_position(1, 0, 2.50)
    settings.add_sensor(camera_RGB)

    camera_seg = Camera('CameraSegmentation', PostProcessing='SemanticSegmentation')
    camera_seg.set_image_size(256, 256)
    camera_seg.set_position(1, 0, 2.50)
    settings.add_sensor(camera_seg)

    return settings


def convert_image(sensor_data, simple_seg=True):
    obs = np.frombuffer(sensor_data['CameraRGB'].raw_data, dtype=np.uint8).reshape((256, 256, 4))[:, :, :3]
    seg = labels_to_array(sensor_data['CameraSegmentation'])
    if simple_seg:
        seg = simplify_seg(seg)
    return obs, seg


def simplify_seg(array):
    classes = {
        0: 0,   # None
        1: 1,   # Buildings
        2: 1,   # Fences
        3: 1,   # Other
        4: 1,   # Pedestrians
        5: 1,   # Poles
        6: 3,   # RoadLines
        7: 3,   # Roads
        8: 2,   # Sidewalks
        9: 1,   # Vegetation
        10: 1,  # Vehicles
        11: 1,  # Walls
        12: 1   # TrafficSigns
    }

    result = np.zeros_like(array, dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


def reward_from_info(info):
    reward = dict()
    reward['without_pos'] = info['speed'] / 15 - info['offroad'] * 2.0 - info['collision']
    reward['with_pos'] = reward['without_pos'] - info['other_lane'] / 2
    return reward


class carla_env(object):
    def __init__(self, client, simple_seg=True):
        self.client = client
        self.simple_seg = simple_seg

    def reset(self, testing=False):
        self.testing = testing
        self.timestep = 0
        self.collision = 0
        self.stuck_cnt = 0
        self.collision_cnt = 0
        self.offroad_cnt = 0
        self.ignite = False
        self.scene = self.client.load_settings(default_settings())
        number_of_player_starts = len(self.scene.player_start_spots)
        player_start = 0  # random.randint(0, max(0, number_of_player_starts - 1))
        print('Starting new episode at %r...' % self.scene.map_name)
        self.client.start_episode(player_start)

        for frame in range(30):
            measurements, sensor_data = self.client.read_data()
            self.client.send_control(
                steer=random.uniform(-1.0, 1.0),
                throttle=0.5,
                brake=0.0,
                hand_brake=False,
                reverse=False)

        measurements, sensor_data = self.client.read_data()
        info = self.convert_info(measurements)
        obs, seg = convert_image(sensor_data, self.simple_seg)
        return (obs, seg), info

    def step(self, action, expert=False):
        self.timestep += 1
        if expert:
            self.client.send_control(action)
        else:
            self.client.send_control(
                steer=action[1] * 0.5,
                throttle=0.5 * action[0] + 0.5,
                brake=0.0,
                hand_brake=False,
                reverse=False)
        measurements, sensor_data = self.client.read_data()
        info = self.convert_info(measurements)
        obs, seg = convert_image(sensor_data, self.simple_seg)
        reward = reward_from_info(info)
        done = self.done_from_info(info) or self.timestep > 1000

        return (obs, seg), reward, done, info

    def convert_info(self, measurements):
        info = dict()
        info['speed'] = measurements.player_measurements.forward_speed
        # print('collision_other', measurements.player_measurements.collision_other)
        # print('collision_pedestrians', measurements.player_measurements.collision_pedestrians)
        # print('collision_vehicles', measurements.player_measurements.collision_vehicles)
        info['collision'] = int(measurements.player_measurements.collision_other + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_vehicles > self.collision or (self.collision_cnt > 0 and info['speed'] < 0.5))
        self.collision = measurements.player_measurements.collision_other + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_vehicles
        info['other_lane'] = measurements.player_measurements.intersection_otherlane
        info['offroad'] = int(measurements.player_measurements.intersection_offroad > 0.001)
        info['expert_control'] = measurements.player_measurements.autopilot_control
        return info

    def done_from_info(self, info):
        if info['collision'] > 0 or (self.collision_cnt > 0 and info['speed'] < 0.5):
            self.collision_cnt += 1
        else:
            self.collision_cnt = 0
        # self.collision_cnt = (self.collision_cnt + info['collision']) * info['collision']
        self.ignite = self.ignite or info['speed'] > 1
        stuck = int(info['speed'] < 1)
        self.stuck_cnt = (self.stuck_cnt + stuck) * stuck * int(bool(self.ignite) or self.testing)

        if info['offroad'] > 0.5:
            self.offroad_cnt += 1

        return (self.stuck_cnt > 30 and self.ignite) or self.offroad_cnt > 30 or self.collision_cnt > 20


if __name__ == '__main__':
    with make_carla_client('localhost', 2000) as client:
        print('CarlaClient connected')
        env = carla_env(client)
        env.reset()
        for i in range(20):
            obs, seg, reward, done, info = env.step([1, 0])
            print(info)
        env.reset()
        for i in range(20):
            obs, seg, reward, done, info = env.step([1, 0])
            print(info)