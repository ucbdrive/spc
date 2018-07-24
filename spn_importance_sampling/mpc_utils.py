from __future__ import division, print_function
import gym
import numpy as np
import random
import torch
import math
import pdb

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

        self.next_idx      = 0
        self.num_in_buffer = 0
        self.ret      = 0

        self.obs      = None
        self.action   = None
        self.done     = None
        self.coll     = None
        self.offroad  = None
        self.pos      = None
        self.angle    = None
        self.speed    = None
        self.seg      = None
        self.xyz      = None
        self.reward   = None
        self.last_done_idx = 0

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
            done_list = self.done[idx - self.args.frame_history_len: idx - self.args.frame_history_len + 1 + self.args.pred_step]
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
        data_dict['sp_batch'] = np.concatenate([self.speed[idx: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)
        
        if self.args.use_collision:
            data_dict['coll_batch'] = np.concatenate([self.coll[idx + 1: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)
        
        if self.args.use_offroad:
            data_dict['off_batch'] = np.concatenate([self.offroad[idx + 1: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)
        
        if self.args.use_pos:
            data_dict['pos_batch'] = np.concatenate([self.pos[idx + 1: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)
        
        if self.args.use_angle or self.args.use_distance:
            data_dict['angle_batch'] = np.concatenate([self.angle[idx: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)
        
        if self.args.use_distance:
            data_dict['dist_batch'] = data_dict['sp_batch'] * (np.cos(data_dict['angle_batch']) - np.abs(np.sin(data_dict['angle_batch'])))
        
        if self.args.use_seg:
            data_dict['seg_batch'] = np.concatenate([self.seg[idx: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)
        
        if self.args.use_xyz:
            data_dict['xyz_batch'] = np.concatenate([self.xyz[idx: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)

        return data_dict

    def sample(self, batch_size, sample_early=False):
        assert self.can_sample(batch_size)
        if sample_early == False:
            idxes = self.sample_n_unique(lambda: np.random.choice(self.num_in_buffer, p=self.reward[:self.num_in_buffer]/self.reward[:self.num_in_buffer].sum()), batch_size)
        else:
            idxes = self.sample_n_unique(lambda: random.randint(0, int(self.num_in_buffer / 3) - 2), batch_size)
        return self._encode_sample(idxes), idxes

    def encode_recent_observation(self):
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.args.buffer_size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
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
            self.obs      = np.empty([self.args.buffer_size] + list(frame.shape),         dtype = np.uint8)
            self.action   = np.zeros([self.args.buffer_size] + [self.args.num_total_act], dtype = np.float32)
            self.done     = np.empty([self.args.buffer_size],                             dtype = np.int32)
            self.reward   = np.empty([self.args.buffer_size],                             dtype = np.int32)

            if self.args.use_collision:
                self.coll = np.empty([self.args.buffer_size] + [1], dtype=np.int32)
            if self.args.use_offroad:
                self.offroad = np.empty([self.args.buffer_size] + [1], dtype=np.int32)

            self.pos      = np.empty([self.args.buffer_size, 1],    dtype=np.float32)
            self.angle    = np.empty([self.args.buffer_size, 1],    dtype=np.float32)
            self.speed    = np.empty([self.args.buffer_size, 1],    dtype=np.float32)

            if self.args.use_seg:
                self.seg  = np.empty([self.args.buffer_size] + [1, 256, 256], dtype=np.uint8)

            if self.args.use_xyz:
                self.xyz  = np.empty([self.args.buffer_size, 3], dtype=np.float32)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.args.buffer_size
        self.num_in_buffer = min(self.args.buffer_size, self.num_in_buffer + 1)
        self.ret = ret
        return ret
    
    def store_action(self, idx, action, done, reward):
        if self.args.continuous:
            self.action[idx, :] = action
        else:
            self.action[idx, int(action)] = 1
        self.done[idx] = int(done)

        if done:
            if self.last_done_idx > idx:
                self.reward[self.last_done_idx:] = reward
                self.reward[:idx+1] = reward
            else:
                self.reward[self.last_done_idx:idx+1] = reward
        self.last_done_idx = idx
        
    def store_effect(self, idx, coll, off, speed, angle, pos, xyz, seg):
        if self.args.use_xyz:
            self.xyz[idx, :] = xyz

        if self.args.use_seg:
            self.seg[idx, :] = seg

        
        if self.args.use_collision:
            self.coll[idx, 0] = int(coll)
        if self.args.use_offroad:
            self.offroad[idx, 0] = int(off)
        self.speed[idx, 0] = speed
        self.angle[idx, 0] = angle
        self.pos[idx, 0] = pos
