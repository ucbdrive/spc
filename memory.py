from __future__ import division, print_function
import numpy as np
import os
import random
import torch
from torch.autograd import Variable
import pickle


class SPCBuffer(object):
    def __init__(self, args):
        self.args = args

        self.next_idx = 0
        self.num_in_buffer = 0
        self.last_idx = 0

        self.obs = None
        self.action = None
        self.done = None
        self.collision = None
        self.offroad = None
        self.speed = None
        self.seg = None
        self.expert = None
        self.guide_action = None
        self.epi_lens = []

    def can_sample_guide(self, batch_size):
        # determines whether there are enough expert data for self-imitation learning
        if len(self.epi_lens) == 0:
            return False
        if self.args.verbose:
            print('Calculating bar from %s' % str(self.epi_lens))
        bar = self.get_bar()
        if self.args.verbose:
            print('Bar: %d' % bar)
        bar_index = np.where(self.expert[:self.num_in_buffer] >= bar)[0]
        if self.args.verbose:
            print('Number of candidates: %d' % len(bar_index))
        return len(bar_index) >= batch_size

    def get_bar(self):
        # calculate the bar according to which expert guidance data are selected
        idx = int(len(self.epi_lens) * self.args.expert_ratio)
        bar = max(sorted(self.epi_lens, reverse=True)[idx], self.args.expert_bar)
        return bar

    def sample_guide(self, batch_size):
        # sample expert guidance replay data for self-imitation learning
        indices = np.where(self.expert[:self.num_in_buffer] >= self.get_bar())[0]
        indices = list(np.random.choice(list(indices), batch_size))
        obs = Variable(torch.from_numpy(np.concatenate([self.obs[idx][np.newaxis, :] for idx in indices], axis=0)).float() / 255.0, requires_grad=False)
        guide_action = Variable(torch.from_numpy(self.guide_action[indices]), requires_grad=False).long()

        if torch.cuda.is_available():
            obs = obs.cuda()
            guide_action = guide_action.cuda()
        return obs, guide_action

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
            done_list = self.done[idx-self.args.frame_history_len+1: idx+self.args.pred_step+1]
            if np.sum(done_list) >= 1.0:
                return False
            else:
                return True

    def can_sample(self, batch_size):
        return batch_size * self.args.pred_step + 1 <= self.num_in_buffer

    def _encode_sample(self, indices):
        data_dict = dict()

        data_dict['obs_batch'] = np.concatenate([np.concatenate([self._encode_observation(idx+ii)[np.newaxis,:] for ii in range(self.args.pred_step)], 0)[np.newaxis, :] for idx in indices], axis=0)
        data_dict['nx_obs_batch'] = np.concatenate([np.concatenate([self._encode_observation(idx+1+ii)[np.newaxis,:] for ii in range(self.args.pred_step)], 0)[np.newaxis, :] for idx in indices], axis=0)
        data_dict['act_batch'] = np.concatenate([self.action[idx: idx+self.args.pred_step, :][np.newaxis, :] for idx in indices], axis=0)
        data_dict['sp_batch'] = np.concatenate([self.speed[idx: idx+self.args.pred_step+1][np.newaxis, :] for idx in indices], axis=0)
        data_dict['prev_action'] = np.concatenate([self.action[idx-self.args.frame_history_len+1: idx, :][np.newaxis, :] for idx in indices], axis=0)
        data_dict['coll_batch'] = np.concatenate([self.collision[idx+1: idx+self.args.pred_step+1][np.newaxis, :] for idx in indices], axis=0)
        data_dict['off_batch'] = np.concatenate([self.offroad[idx+1: idx+self.args.pred_step+1][np.newaxis, :] for idx in indices], axis=0)
        data_dict['seg_batch'] = np.concatenate([self.seg[idx: idx+self.args.pred_step+1, :][np.newaxis, :] for idx in indices], axis=0)

        return data_dict

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        indices = self.sample_n_unique(lambda: random.randint(10, self.num_in_buffer - 10), batch_size)
        return self._encode_sample(indices)

    def _encode_observation(self, idx):
        start_idx = idx - self.args.frame_history_len + 1
        end_idx = idx + 1
        assert start_idx >= 0 and end_idx <= min(self.num_in_buffer, self.args.buffer_size) and np.sum(self.done[start_idx: end_idx]) == 0
        return self.obs[start_idx: end_idx].reshape(-1, self.args.frame_height, self.args.frame_width)

    def store_frame(self, obs, collision, offroad, speed, seg):
        assert obs.shape == (self.args.frame_height, self.args.frame_width, 3)
        frame = obs.transpose(2, 0, 1)  # reshape as [C, H, W]

        if self.obs is None:
            self.obs = np.empty([self.args.buffer_size, 3, self.args.frame_height, self.args.frame_width], dtype=np.uint8)
            self.action = np.empty([self.args.buffer_size, self.args.num_total_act], dtype=np.float32)
            self.done = np.empty([self.args.buffer_size], dtype=np.int32)
            self.expert = np.empty([self.args.buffer_size], dtype=np.float32)
            self.guide_action = np.empty([self.args.buffer_size], dtype=np.int32)
            self.collision = np.empty([self.args.buffer_size], dtype=np.int32)
            self.offroad = np.empty([self.args.buffer_size], dtype=np.int32)
            self.speed = np.empty([self.args.buffer_size], dtype=np.float32)
            self.seg = np.empty([self.args.buffer_size, self.args.frame_height, self.args.frame_width], dtype=np.uint8)

        self.obs[self.next_idx] = frame
        self.collision[self.next_idx] = int(collision)
        self.offroad[self.next_idx] = int(offroad)
        self.speed[self.next_idx] = speed
        self.seg[self.next_idx, :] = seg

        self.last_idx = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.args.buffer_size
        self.num_in_buffer = min(self.args.buffer_size, self.num_in_buffer + 1)

    def store_action(self, guide_action, action, done):
        self.guide_action[self.last_idx] = guide_action
        self.action[self.last_idx, :] = action
        self.done[self.last_idx] = int(done)

    def load(self, path):
        if os.path.exists(os.path.join(path, 'spc_buffer.pkl')):
            with open(os.path.join(path, 'spc_buffer.pkl'), 'rb') as f:
                self.__dict__ = pickle.load(f)

    def save(self, path):
        with open(os.path.join(path, 'spc_buffer.pkl'), 'wb') as f:
            pickle.dump(self.__dict__, f)
