import numpy as np
import re
import random
import math
import os
import time
from .sensor import CollisionSensor, LaneInvasionSensor, CameraManager
import cv2
import shutil
import sys
import glob

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def world_env(args):
    client = carla.Client('localhost', args.port)
    client.set_timeout(20.0)
    carla_world = client.get_world()
    # by default, spc only supports running on carla in sync mode
    settings = carla_world.get_settings()
    settings.synchronous_mode = True
    client.get_world().apply_settings(settings)
    env = World(args, carla_world)

class World(object):
    def __init__(self, args, carla_world):
        print("Begin to iniltialize the world")
        self.args = args
        self.world = carla_world
        self.map = self.world.get_map()
        self.player = None
        self.word_port = self.args.port
        # self.clock = pygame.time.Clock()
        self.ts = 0
        self.frame = 0
        
        self._weather_presets = find_weather_presets()
        self._weather_index = self.args.weather_id
        self.vehicle_num = self.args.vehicle_num
        self.ped_num = self.args.ped_num

        # sensors and camera
        self.rgb_camera = None
        self.seg_camera = None
        self.col_sensor = None
        self.offroad_detector = None

        # global environments
        self.vehicles = []
        self.peds = []
        self.timestamp = 0
        self.episode = 0
        self.log = open(os.path.join(self.args.save_path, 'action_info_log.txt'), 'w')

        # variables for training policy
        self.collision = False
        self.offroad = False
        self.offlane = False
        self.stuck_cnt = 0
        self.collision_cnt = 0
        self.offroad_cnt = 0
        self.ignite = False
        self.reward = 0.0
        self.crosssolid_cnt = 0

        # initialize the recording configs
        self.recording_enabled = self.args.recording
        self.recording_start = 0
    
        # start the first episode
        self.reset()

    def find_recording_dir(self, root_dir):
        record_dir = os.path.join(root_dir, str(self.episode))
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        self.args.record_dir = record_dir
        
    def setup_player(self):
        # set up a vehicle player as the main agent 
        blueprint_library = self.world.get_blueprint_library()
        vehicles = self.world.get_blueprint_library().filter('vehicle.ford.mustang')
        vehicle_bp = vehicles[0]
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points[:10]) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point) 

    def setup_sensor(self):
        # set up the camera to observe surrounding
        self.rgb_camera = CameraManager(self.player, self.args, tag="RGB")
        self.seg_camera = CameraManager(self.player, self.args, tag="SEG")
        self.rgb_camera.set_sensor(0)
        self.seg_camera.set_sensor(5)
        self.col_sensor = CollisionSensor(self.player)
        self.offroad_detector = LaneInvasionSensor(self.player)

        # monitoring the vehicle driving
        self.monitor = CameraManager(self.player, self.args, x=-5.5, y=0, z=2.8, monitor=True, tag="MONITOR")
        self.monitor.set_sensor(0)
    
    def info(self):
        v = self.player.get_velocity()
        self.collision = self.col_sensor.check()
        self.offlane, self.offroad  = self.offroad_detector.check()

        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        location = self.player.get_location()
        waypoint = self.map.get_waypoint(location, project_to_road=False)
        info = {}
        info["speed"] = speed
        info["collision"] = 1 if (self.collision or (self.collision_cnt > 0 and speed < 0.5)) else 0 
        info["other_lane"] = 1 if self.offlane else 0
        # Only in carla later than 0.9.4, the python api to return the lane
        # type is provided, which can be used to detect offroad.
        if waypoint is not None and waypoint.lane_type == "driving":
            info["offroad"] = 0
        else:
            info["offroad"] = 1
        info["expert_control"] = self.args.autopilot
        info["seg"] = self.seg_camera.observe()
        return info
    
    def observe(self):
        return self.rgb_camera.observe()

    def setup_road(self, vehicle_num, ped_num):
        for i in range(vehicle_num):
            pass

        for i in range(ped_num):
            pass

    def record(self):
       #  time.sleep(0.3)
        if self.args.recording_frame:
            self.rgb_camera.save(self.timestamp)
            self.seg_camera.save(self.timestamp)
        if self.args.monitor:
            self.monitor.save(self.timestamp)
    
    def monitor_video(self):
        print("produce monitor video of episode: {}".format(self.episode))
        img_dir = os.path.join(self.args.record_dir, "monitor")
        fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
        img_list = os.listdir(img_dir)
        # num = int(img_list[-1].split('.')[0])
        num = self.timestamp
        video_dir = os.path.join(self.args.record_dir, "monitor_{}.avi".format(num))
        vw = cv2.VideoWriter(video_dir, fourcc, 24, \
            (self.args.frame_width, self.args.frame_height))
        for i in range(1, num+1):
            try:
                frame = cv2.imread(os.path.join(img_dir, "%08d.png" % i))
            except:
                continue
            vw.write(frame)
        vw.release()
        shutil.rmtree(img_dir)

    def reset(self, testing=False):
        print('Starting new episode ...')
        if self.args.monitor and self.episode > 0:
            # save the trianing monitoring images of last episode into video
            self.monitor_video()

        self.testing = testing
        self.episode += 1
        self.crosssolid_cnt = 0
        self.timestamp = 0
        self.collision = 0
        self.stuck_cnt = 0
        self.collision_cnt = 0
        self.offroad_cnt = 0
        self.ignite = False
        self.log.flush()
        self.log.write("\n\n=============== episode: {} =============== \n\n".format(self.episode))

        # set the director for recording of each frame
        if self.args.recording_frame or self.args.monitor:
            self.find_recording_dir(os.path.join(self.args.save_path, self.args.monitor_video_dir))

        if self.player is not None:
            self.destroy()
            self.player = None
        
        # setup the POV player and bind sensor/cameras
        while self.player is None:
            self.setup_player()
            self.setup_sensor()

        self.setup_road(self.vehicle_num, self.ped_num)

        time.sleep(2)
        self.tick()

        for frame in range(200):
            self.timestamp += 1
            self.record()
            control = self.player.get_control()
            control.steer = random.uniform(-1.0, 1.0)
            control.throttle = 0.6
            control.brake = 0.0
            hand_brak = False
            reverse = False
            self.player.apply_control(control)

            # self.clock.tick()
            self.tick()
            # self.ts = self.world.wait_for_tick(seconds=100.0)
            # if self.ts.frame_count != self.frame + 1:
            #     self.log.write("reset: -------- a frame lost -------------: {} {}\n".format(self.ts.frame_count, self.frame))
            # self.frame = self.ts.frame_count

        info = self.info()
        obs = self.observe()
        self.log.write("step {}: speed: {} | collision: {} | offroad: {} | other_lane: {}\n".format(self.timestamp, round(info['speed'], 5), info['collision'], info['offroad'], info['other_lane']))

        return obs, info

    def step(self, action, expert=False):
        # 1. set and send control signal
        throttle, steer = action
        reverse = False
        brake = 0
        hand_brake = False
        control = self.player.get_control()
        if expert:
            control.throttle = throttle
            control.reverse = reverse
            control.steer = steer
            control.brake = brake
            control.hand_brake = hand_brake
            self.player.apply_control(control)
        else:
            control.throttle = throttle*0.5 + 0.5
            control.steer = steer*0.25
            control.brake = 0
            control.hand_brake = False
            control.reverse = False
            self.player.apply_control(control)
        
        # 2. return current information
        # self.clock.tick()
        self.tick()
        # self.ts = self.world.wait_for_tick(seconds=100.0)
        # if self.ts.frame_count != self.frame + 1:
        #     self.log.write("step: --------------- a frame lost -------------: {} {}\n".format(self.ts.frame_count, self.frame))
        # self.frame = self.ts.frame_count

        obs = self.observe()
        info = self.info()
        # print("========== rgb frame {} seg frame {}=======".format(rgb_frame, seg_frame))
        reward = self.reward_from_info(info)
        done = self.done_from_info(info) or self.timestamp > 1000
        if done:
            if self.stuck_cnt > 30 and self.timestamp > 50:
                self.log.write("done because of stuck\n")
            if self.offroad_cnt > 30:
                self.log.write("done because of offroad\n")
            if self.collision_cnt > 20:
                self.log.write("done because of collision\n")
            if self.timestamp > 1000:
                self.log.write("done because of timestamp out\n")

        # 3. record frame
        self.timestamp += 1
        self.record()

        self.log.write("step {}: speed: {} | collision: {} | offroad: {} | other_lane: {}\n".format(self.timestamp, round(info['speed'], 5), info['collision'], info['offroad'], info['other_lane']))
        
        return obs, reward, done, info, self.timestamp


    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock=None):
        frame = self.world.tick()
        # print("======== tick frame: {} ===========".format(frame))

    def render(self, display):
        self.camera_manager.render(display)

    def reward_from_info(self, info):
        reward = dict()
        reward['without_pos'] = info['speed'] / 15 - info['offroad'] - info['collision'] * 2.0
        reward['with_pos'] = reward['without_pos'] - info['other_lane'] / 2
        return reward['without_pos']

    def done_from_info(self, info):
        if info['collision'] > 0 or (self.collision_cnt > 0 and info['speed'] < 0.5):
            self.collision_cnt += 1
        else:
            self.collision_cnt = 0

        self.ignite = self.ignite or info['speed'] > 1
        stuck = int(info['speed'] < 1)
        self.stuck_cnt = (self.stuck_cnt + stuck) * stuck * int(bool(self.ignite) or self.testing)

        if info['offroad'] == 1:
            self.offroad_cnt += 1
        return (self.stuck_cnt > 30 and self.timestamp > 50) or self.collision_cnt > 20 or self.offroad_cnt > 30

    def destroySensors(self):
        self.rgb_camera.sensor.destroy()
        self.rgb_camera = None
        self.seg_camera.sensor.destroy()
        self.seg_camera = None
        self.monitor.sensor.destroy()
        self.monitor = None

    def destroy(self):
        actors = [
            self.rgb_camera.sensor,
            self.seg_camera.sensor,
            self.col_sensor.sensor,
            self.offroad_detector.sensor,
            self.monitor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()
                actor = None
        
