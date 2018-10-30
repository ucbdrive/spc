import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

def action2id(action):
    if action[0] < 0:
        return 3
    if np.abs(action[1]) < 0.02:
        return 0
    if action[1] < -0.02:
        return 1
    if action[1] > 0.02:
        return 2

class CarlaDataset(Dataset):
    def __init__(self, path=os.path.join('..', 'imitation_data')):
        super(CarlaDataset, self).__init__()
        self.path = []
        for dir_l1 in os.listdir(path):
            cwd = os.path.join(path, dir_l1)
            for dir_l2 in os.listdir(cwd):
                self.path.append(os.path.join(cwd, dir_l2))

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        path = self.path[item]
        img = torch.from_numpy(cv2.imread(os.path.join(path, 'obs.png')).transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(cv2.imread(os.path.join(path, 'seg.png'), 0)).long()
        action = action2id(np.load(os.path.join(path, 'action.npy')))
        action = torch.tensor(action).long()
        return img, mask, action

