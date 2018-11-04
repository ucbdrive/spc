from __future__ import division, print_function
import numpy as np
import os
import json
import random
import torch
from torch.autograd import Variable
import math
import pdb
from sklearn.preprocessing import OneHotEncoder


class IMGBuffer(object):
    def __init__(self):
        self.cnt = 0
        self.avg = 0
        self.std = 0

    def store_frame(self, frame):
        avg = frame.mean()
        std = frame.std()
        self.avg = (self.cnt * self.avg + avg) / (self.cnt + 1)
        self.std = math.sqrt((self.cnt * self.std ** 2 + std ** 2) / (self.cnt + 1))
        self.cnt += 1

    def get_avg_std(self):
        return self.avg, self.std


class MPCBuffer(object):
    def __init__(self, args):
        self.args = args

        self.next_idx = 0
        self.num_in_buffer = 0
        self.ret = 0

        self.obs = None
        self.action = None
        self.done = None
        self.seg = None
        self.reward = None
        self.coll = None

    def sample_n_unique(self, sampling_f, n):
        res = []
        while len(res) < n:
            candidate = sampling_f()
            done = self.sample_done(candidate)
            if candidate not in res and done:
                res.append(candidate)
        return res

    def sample_done(self, idx):
        if idx < 10 or idx >= self.num_in_buffer - self.args.pred_step - 10:
            return False
        else:
            done_list = self.done[idx - self.args.frame_history_len: idx + self.args.pred_step - 1]
            if np.sum(done_list) >= 1.0:
                return False
            else:
                return True

    def can_sample(self, batch_size):
        return batch_size * self.args.pred_step + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        data_dict = dict()

        data_dict['obs_batch'] = np.concatenate([np.concatenate([self._encode_observation(idx+ii)[np.newaxis,:] for ii in range(self.args.pred_step)], 0)[np.newaxis,:] for idx in idxes], 0)
        data_dict['nx_obs_batch'] = np.concatenate([np.concatenate([self._encode_observation(idx+1+ii)[np.newaxis,:] for ii in range(self.args.pred_step)], 0)[np.newaxis,:] for idx in idxes], 0)
        data_dict['act_batch'] = np.concatenate([self.action[idx: idx + self.args.pred_step, :][np.newaxis, :] for idx in idxes], 0)
        data_dict['reward_batch'] = np.concatenate([self.reward[idx: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)
        data_dict['value_batch'] = np.concatenate([self.value[idx + 1: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)
        data_dict['coll_batch'] = np.concatenate([self.coll[idx + 1: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)

        if self.args.lstm2:
            data_dict['prev_action'] = np.concatenate([self.action[idx-self.args.frame_history_len+1: idx, :][np.newaxis, :] for idx in idxes], 0)
        else:
            data_dict['prev_action'] = None

        if self.args.use_seg:
            data_dict['seg_batch'] = np.concatenate([self.seg[idx: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)

        return data_dict

    def sample(self, batch_size, sample_early=False):
        assert self.can_sample(batch_size)
        if not sample_early:
            idxes = self.sample_n_unique(lambda: random.randint(10, self.num_in_buffer - 10), batch_size)
        else:
            idxes = self.sample_n_unique(lambda: random.randint(10, int(self.num_in_buffer / 3) - 10), batch_size)
        return self._encode_sample(idxes), idxes

    def encode_recent_observation(self):
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.args.buffer_size)

    def _encode_observation(self, idx):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.args.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.args.buffer_size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.args.buffer_size]:
                start_idx = idx + 1
        missing_context = self.args.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.args.buffer_size])
            return np.concatenate(frames, 0)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame):
        if len(frame.shape) > 1:
            # transpose image frame into (img_c, img_h, img_w)
            frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs = np.empty([self.args.buffer_size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.args.buffer_size, self.args.num_total_act], dtype=np.float32)
            self.done = np.empty([self.args.buffer_size], dtype=np.uint8)
            self.coll = np.empty([self.args.buffer_size, 1], dtype=np.uint8)

            self.reward = np.empty([self.args.buffer_size, 1], dtype=np.float32)
            self.value = np.empty([self.args.buffer_size, 1], dtype=np.float32)

            if self.args.use_seg:
                self.seg = np.empty([self.args.buffer_size] + [1, self.args.frame_height, self.args.frame_width], dtype=np.uint8)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.args.buffer_size
        self.num_in_buffer = min(self.args.buffer_size, self.num_in_buffer + 1)
        self.ret = ret
        return ret

    def store_action(self, idx, action, done):
        if self.args.continuous:
            self.action[idx, :] = action
        else:
            self.action[idx, int(action)] = 1
        self.done[idx] = int(done)

    def store_effect(self, idx, reward=None, seg=None):
        self.reward[idx, 0] = reward
        self.coll[idx, 0] = int(reward < 0)

        if self.args.use_seg:
            self.seg[idx, :] = seg

    def load(self, path):
        return
        path = os.path.join(path, 'MPCBuffer')
        try:
            assert os.path.isdir(path)

            assert os.path.exists(os.path.join(path, 'obs.npy'))
            assert os.path.exists(os.path.join(path, 'action.npy'))
            assert os.path.exists(os.path.join(path, 'done.npy'))

            assert os.path.exists(os.path.join(path, 'reward.npy'))

            assert os.path.exists(os.path.join(path, 'mpc_buffer.json'))

            self.obs = np.load(os.path.join(path, 'obs.npy'))
            self.action = np.load(os.path.join(path, 'action.npy'))
            self.done = np.load(os.path.join(path, 'done.npy'))
            self.reward = np.load(os.path.join(path, 'reward.npy'))

            if self.args.use_seg:
                self.seg = np.load(os.path.join(path, 'seg.npy'))

            with open(os.path.join(path, 'mpc_buffer.json'), "r") as f:
                state_dict = json.load(f)
            self.idx = state_dict['next_idx']
            self.num_in_buffer = state_dict['num_in_buffer']

        except:
            print('\033[1;31mUnable to load saved MPCBuffer!\033[0m')

    def save(self, path):
        return
        path = os.path.join(path, 'MPCBuffer')
        if not os.path.isdir(path):
            os.mkdir(path)

        np.save(os.path.join(path, 'obs.npy'), self.obs)
        np.save(os.path.join(path, 'action.npy'), self.action)
        np.save(os.path.join(path, 'done.npy'), self.done)

        np.save(os.path.join(path, 'reward.npy'), self.reward)
        if self.args.use_seg:
            np.save(os.path.join(path, 'seg.npy'), self.seg)

        state_dict = {'next_idx': self.next_idx, 'num_in_buffer': self.num_in_buffer}
        with open(os.path.join(path, 'mpc_buffer.json'), "w") as f:
            json.dump(state_dict, f, indent=4)
