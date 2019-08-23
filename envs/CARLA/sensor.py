import os
import carla
import weakref
from carla import ColorConverter as cc
import numpy as np 
import math
import cv2

VIEW_FOV = 90

'''
Semantic Segmentation Value of Color in Carla 0.9.x
Value       Tag         Converted Color
0	    Unlabeled	    ( 0, 0, 0)
1	    Building	    ( 70, 70, 70)
2	    Fence	        (190, 153, 153)
3	    Other	        (250, 170, 160)
4	    Pedestrian	    (220, 20, 60)
5	    Pole	        (153, 153, 153)
6	    Road line	    (157, 234, 50)
7	    Road	        (128, 64, 128)
8	    Sidewalk	    (244, 35, 232)
9	    Vegetation	    (107, 142, 35)
10	    Car	            ( 0, 0, 142)
11	    Wall	        (102, 102, 156)
12	    Traffic sign	(220, 220, 0)
'''

def simplify_seg(array):
    classes = {
        0: 0,   # None
        1: 1,   # Buildings
        2: 1,   # Fences
        3: 1,   # Other
        4: 1,   # Pedestrians
        5: 1,   # Poles
        6: 4,   # RoadLines
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

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name

def convert_image(sensor_data, simple_seg=True):
    obs = np.frombuffer(sensor_data['CameraRGB'].raw_data, dtype=np.uint8).reshape((256, 256, 4))[:, :, :3]
    seg = labels_to_array(sensor_data['CameraSegmentation'])
    if simple_seg:
        seg = simplify_seg(seg)
    return obs, seg

class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        self.last_col_cnt = 0
        self.col_cnt = 0

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    def check(self):
        if self.col_cnt == self.last_col_cnt:
            return False
        else:
            self.last_col_cnt = self.col_cnt
            print(self.col_log)
            return True

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        self.col_cnt += 1
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.col_log = "Collision with %r" % actor_type
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        # bp = world.get_blueprint_library().find('sensor.other.lane_invasion') # to support carla0.9.5
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
        self.last_offlane_cnt = 0
        self.offlane_cnt = 0
        self.type_flag = 0

    def check(self):
        if self.offlane_cnt == self.last_offlane_cnt:
            return False, False
        else:
            self.last_offlane_cnt = self.offlane_cnt
            return self.type_flag == 1, self.type_flag == 2

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        self.offlane_cnt += 1
        if not self:
            return
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        if text[0] == "'Broken'":
            self.type_flag = 1
        elif text[0] == "'Solid'":
            self.type_flag = 2
        else:
            exit(0)


class CameraManager(object):
    def __init__(self, parent_actor, args, hud=None, x=1, y=0, z=2.5, monitor=False, tag=None):
        self.args = args
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = self.args.recording_frame
        self.notify = args.notify
        self.monitor = monitor
        self.monitor_record = self.args.monitor
        self.tag = tag
        self.output = np.zeros([args.frame_height, args.frame_width, 3])
        self.timestamp = -1
        self.camera_transform = carla.Transform(carla.Location(x, y, z))
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(self.args.frame_width))
                bp.set_attribute('image_size_y', str(self.args.frame_height))
                bp.set_attribute('fov', str(VIEW_FOV))
            item.append(bp)
        self._index = None
        self.record_image = None
            
        # set directories for realtime frame recording if needed
        if self.args.recording_frame:
            if not os.path.isdir(os.path.join(self.args.record_dir, "obs")):
                os.makedirs(os.path.join(self.args.record_dir, "obs"))

        if self.args.monitor:
            if not os.path.isdir(os.path.join(self.args.record_dir, "monitor")):
                os.makedirs(os.path.join(self.args.record_dir, "monitor"))

    def toggle_camera(self):
        self.sensor.set_transform(self.camera_transform)

    def set_sensor(self, index):
        index = index % len(self._sensors)
        if self._index is None:
            needs_respawn = True
        else:
            needs_respawn = self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self.camera_transform,
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular reference.
            weak_self = weakref.ref(self)
            if index == 0 and not self.monitor:
                # for RGB camera
                self.sensor.listen(lambda image: CameraManager._parse_rgb_image(weak_self, image))
            elif index == 0 and self.monitor:
                self.sensor.listen(lambda image: CameraManager._parse_monitor_image(weak_self, image))
            elif index == 5:
                self.output = np.zeros([self.args.frame_height, self.args.frame_width, 1])
                self.sensor.listen(lambda image: CameraManager._parse_seg_image(weak_self, image))
            else:
                assert(0) # currently we only these the three sensros above
        if self.notify and not self._hud:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        if not self._hud:
            self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    def observe(self):
        return self.output

    def save(self, timestamp):
        self.timestamp = timestamp
        save_dir = self.args.record_dir
        if self.tag == "RGB":
            if self.record_image is not None:
                self.record_image.save_to_disk(os.path.join(save_dir, "obs/{}.jpg".format(timestamp)))
        elif self.tag == "SEG":
            if self.record_image is not None:
                self.record_image.save_to_disk(os.path.join(save_dir, "segs/{}.jpg".format(timestamp)))
        elif self.tag == "MONITOR":
            if self.record_image is not None:
                record_name = os.path.join(self.args.record_dir, "monitor/%08d.png" % self.timestamp)
                self.record_image.save_to_disk(record_name)
        else:
            assert(0)

    @staticmethod
    def _parse_rgb_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self._sensors[self._index][1])
        if self._recording:
            record_name = os.path.join(self.args.record_dir, "obs/%08d.png" % image.frame_number)
            image.save_to_disk(record_name)
        self.record_image = image
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.output = array

    @staticmethod
    def _parse_monitor_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self._sensors[self._index][1])
        self.record_image = image

    @staticmethod
    def _parse_seg_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3] # RGBA -> RGB
        ori_array = array
        array = array[:, :, -1] # Only last channel, for segmentation, it's class type
        array = simplify_seg(array)
        ori_array = array
        ori_array *= 50
        cv2.imwrite("sim_seg/%8d.png" % image.frame_number, ori_array)
        image.convert(self._sensors[self._index][1])
        self.record_image = image
        self.output = array
        
