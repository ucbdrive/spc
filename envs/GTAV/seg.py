from functools import partial
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from .dataset.transform import SegmentationTransform

import envs.GTAV.models as models
from .modules.bn import ABN
from .modules.deeplab import DeeplabV3
from utils import color_text


class_map = {
    0: 2,  # "animal--bird"
    1: 2,  # "animal--ground-animal"
    2: 0,  # "construction--barrier--curb"
    3: 0,  # "construction--barrier--fence"
    4: 0,  # "construction--barrier--guard-rail"
    5: 0,  # "construction--barrier--other-barrier"
    6: 0,  # "construction--barrier--wall"
    7: 1,  # "construction--flat--bike-lane"
    8: 1,  # "construction--flat--crosswalk-plain"
    9: 1,  # "construction--flat--curb-cut"
    10: 1, # "construction--flat--parking"
    11: 0, # "construction--flat--pedestrian-area"
    12: 1, # "construction--flat--rail-track"
    13: 1, # "construction--flat--road"
    14: 1, # "construction--flat--service-lane"
    15: 0, # "construction--flat--sidewalk"
    16: 0, # "construction--structure--bridge"
    17: 0, # "construction--structure--building"
    18: 0, # "construction--structure--tunnel"
    19: 2, # "human--person"
    20: 2, # "human--rider--bicyclist"
    21: 2, # "human--rider--motorcyclist"
    22: 2, # "human--rider--other-rider"
    23: 1, # "marking--crosswalk-zebra"
    24: 1, # "marking--general"
    25: 0, # "nature--mountain"
    26: 0, # "nature--sand" Ignored, due to rare to see
    27: 3, # "nature--sky"
    28: 0, # "nature--snow" Not sure whether snow mountain or snow on road
    29: 0, # "nature--terrain" Ignored due to rare appearance
    30: 0, # "nature--vegetation"
    31: 0, # "nature--water"
    32: 0, # "object--banner"
    33: 0, # "object--bench"
    34: 0, # "object--bike-rack"
    35: 0, # "object--billboard"
    36: 0, # "object--catch-basin"  Ignored since not frequent
    37: 0, # "object--cctv-camera"  Ignored since not frequent
    38: 0, # "object--fire-hydrant"
    39: 0, # "object--junction-box"
    40: 0, # "object--mailbox"
    41: 0, # "object--manhole"
    42: 0, # "object--phone-booth"
    43: 0, # "object--pothole" Ignored, since not frequent
    44: 0, # "object--street-light"
    45: 0, # "object--support--pole"
    46: 0, # "object--support--traffic-sign-frame"
    47: 0, # "object--support--utility-pole"
    48: 0, # "object--traffic-light"
    49: 0, # "object--traffic-sign--back"
    50: 0, # "object--traffic-sign--front"
    51: 0, # "object--trash-can"
    52: 2, # "object--vehicle--bicycle"
    53: 0, # "object--vehicle--boat" Ignoring boat
    54: 2, # "object--vehicle--bus"
    55: 2, # "object--vehicle--car"
    56: 2, # "object--vehicle--caravan"
    57: 2, # "object--vehicle--motorcycle"
    58: 2, # "object--vehicle--on-rails"
    59: 2, # "object--vehicle--other-vehicle"
    60: 2, # "object--vehicle--trailer"
    61: 2, # "object--vehicle--truck"
    62: 2, # "object--vehicle--wheeled-slow"
    63: 2, # "void--car-mount"
    64: 2 # "void--ego-vehicle"
}


def vis(array):
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }

    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


class SegmentationModule(nn.Module):
    def __init__(self, head_channels, classes, snapshot_file='model.pth.tar'):
        super(SegmentationModule, self).__init__()

        norm_act = partial(ABN, activation="leaky_relu", slope=.01)
        self.body = models.__dict__["net_wider_resnet38_a2"](norm_act=norm_act, dilation=(1, 2, 4, 4))
        self.head = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))
        self.cls = nn.Conv2d(head_channels, classes, 1)
        self.transform = SegmentationTransform(
            2048,
            (0.41738699, 0.45732192, 0.46886091),
            (0.25685097, 0.26509955, 0.29067996),
        )

        dir_path = os.path.dirname(os.path.realpath(__file__))
        snapshot_file = os.path.join(dir_path, snapshot_file)
        if snapshot_file is not None:
            if not os.path.exists(snapshot_file):
                print(color_text('No local model found at {}'.format(snapshot_file), 'red'))
                print(color_text('Please download pretrained model from https://drive.google.com/file/d/1SJJx5-LFG3J3M99TrPMU-z6ZmgWynxo-/view', 'red'))
            data = torch.load(snapshot_file)
            self.body.load_state_dict(data["state_dict"]["body"])
            self.head.load_state_dict(data["state_dict"]["head"])
            self.cls.load_state_dict(data["state_dict"]["cls"])
            print('Loading segmentation model from %s' % snapshot_file)

    def forward(self, x):
        x = self.transform(x).unsqueeze(0).cuda()
        img_shape = x.shape[-2:]
        x = self.body(x)
        x = self.head(x)
        x = self.cls(x)
        x = F.interpolate(x, size=img_shape, mode='bilinear', align_corners=True)
        x = torch.argmax(x, dim=1).data.cpu().numpy()[0]

        result = np.zeros_like(x, dtype=np.int32)
        for key, value in class_map.items():
            result[np.where(x == key)] = value
        return result


def main():
    cudnn.benchmark = True
    model = SegmentationModule(256, 65)
    model = model.cuda().eval()

    transformation = SegmentationTransform(
        2048,
        (0.41738699, 0.45732192, 0.46886091),
        (0.25685097, 0.26509955, 0.29067996),
    )

    # Run testing
    for fname in os.listdir('a'):
        print(fname)
        x = cv2.imread(os.path.join('a', fname))
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            y = model(x)
        y = vis(y)
        cv2.imwrite(fname.replace('.jpg', '.png'), y)
        break

if __name__ == '__main__':
    main()

