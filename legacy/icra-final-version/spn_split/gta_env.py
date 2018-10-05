"""
GTA V environment

Actions consist of 3 sub-actions:
    - Throttle: 0, 1
    - Brake: 0, 1
    - Steer Left: 0, 1
    - Steer Right: 0, 1

An episode ends when:
    - The agent reaches the pre-fixed duration defined by the user

Reward schedule:
    no collision: 1
    collision: -5
"""

import time
from queue import Queue

from gym import spaces
import numpy as np

from deepgtav.client import Client
from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset

# Constants
NUM_ACTIONS = 3
HOST = 'localhost'  # The IP where DeepGTAV is running
PORT = 8888  # The port where DeepGTAV is running
DRIVING_MODE = -1   # Manual driving
SCREEN_H = 800
SCREEN_W = 1280

# Vehicles
VEHICLES = ['blista', 'voltic', 'packer', 'oracle']

# Weather
WEATHERS = {"CLEAR": 5,
            "EXTRASUNNY": 2,
            "CLOUDS": 5,
            "RAIN": 1,
            "CLEARING": 1,
            "THUNDER": 1,
            "OVERCAST": 0,
            "SMOG": 0,
            "FOGGY": 0,
            "XMAS": 0,
            "SNOWLIGHT": 0,
            "BLIZZARD": 0,
            "NEUTRAL": 0,
            "SNOW": 0}

# Location range
with open("sampled_path_points.np", "rb") as f:
    x_list, y_list = np.load(f)


class GtaEnv():

    def __init__(self, seed=False, duration=1000, vehicle='oracle', weather='CLEAR', time=[10, 30], location=[2000, -609], destination = [-1,-1,-1], autodrive = None):
        """
        GTA V Environment Initiation
        :param seed: [boolean] whether to randomly initiate a scenario
        :param duration: [second] duration of each episode
        :param vehicle: [string] type of vehicles
        :param weather: [string] type of weathers
        :param time: [array] starting time (hour, minute)
        :param location: [array] starting position
        """
        # Env Parameters
        self.direction = destination
        if not (self.direction is None):
            if self.direction == [-1,-1,-1]:
                rand_loc_idx = np.random.randint(len(x_list))
                self.direction = [x_list[rand_loc_idx], y_list[rand_loc_idx]]
        self.driving = autodrive

        if seed:
            self.seed()
        else:
            self.vehicle = vehicle
            self.weather = weather
            self.time = time
            self.location = location
        self.duration = duration  # Number of steps for each episode
        self.done = False

        # Set up connection to DeepGTAV
        print("==== Initiate New GTA Client ====")
        self.client = Client(ip=HOST, port=PORT)
        self.scenario = Scenario(drivingMode=self.driving, vehicle=self.vehicle, weather=self.weather, time=self.time,
                                 location=self.location)
        self.dataset = Dataset(rate=30, frame=[SCREEN_W,SCREEN_H], vehicles=True, peds=True, trafficSigns=True, direction=self.direction, reward=[15.0,0.5],
                 throttle=True, brake=True, steering=True, speed=True, yawRate=True, drivingMode=True, location=True,
                 time=True, roadinfo = True)

        # Action, Observation
        self.action_space = spaces.Box(np.array([0.0,0.0,-1.0]),np.array([1.0,1.0,1.0]))  # throttle, brake, steer
        self.allowed_actions = list(range(NUM_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_H, SCREEN_W, 3))

    def reset(self):

        print("==== Start New Episode ====")
        if self.done:
            self.client.sendMessage(Stop())
        self.done = False
        self.reward = 0.0
        self.curr_ob = None
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration

        self.client.sendMessage(Start(scenario=self.scenario,dataset=self.dataset))
        time.sleep(5)  # IMPORTANT! Wait for the scene to be ready,
                        # if waiting time is too short, client will fail listen to following messages
        message = self.client.recvMessage()
        start_ob = frame2numpy(message['frame'], (SCREEN_W, SCREEN_H))
        message.pop('frame')

        return start_ob, message

    def step(self, action):
        #(throttle, brake, steering)
        if self.driving is None:
            self.client.sendMessage(Commands(float(action[0]), float(action[1]), float(action[2])))

        # Compute new reward and observation
        message = self.client.recvMessage()
        self.curr_ob = frame2numpy(message['frame'], (SCREEN_W, SCREEN_H))
        if time.time() >= self.end_time:
            self.done = True
        message.pop('frame')
        return self.curr_ob, message, self.done

    def seed(self):
        self.vehicle = VEHICLES[np.random.randint(0, len(VEHICLES))]
        weather_list = []
        for k, v in WEATHERS.items():
            weather_list += v * [k]
        self.weather = weather_list[np.random.randint(0, len(weather_list))]
        self.time = [np.random.randint(5, 18), np.random.randint(0, 60)]  # HH, MM
        rand_loc_idx = np.random.randint(len(x_list))
        self.location = [x_list[rand_loc_idx], y_list[rand_loc_idx]]

    def close(self):
        self.client.sendMessage(Stop())
        self.client.close()
