from __future__ import division, print_function
import numpy as np
import torch.multiprocessing as mp
import ctypes
from torch.multiprocessing import Process, Lock, Value, Array

class memory_pool(object):
    def __init__(self, args):
        super(memory_pool, self).__init__()
        self.size = args.buffer_size
        self.frame_history_len = args.frame_history_len
        self.pred_step = args.pred_step # number of prediction steps
        self.num_actions = args.num_total_act

        self.num_in_buffer = Value("i", 0)
        self.next_idx = Value("i", 0)

        self.obs_base = Array(ctypes.c_double, self.size * args.observation_channels * args.observation_height * args.observation_width)
        self.obs = np.ctypeslib.as_array(self.obs_base.get_obj()).reshape((self.size, args.observation_channels, args.observation_height, args.observation_width))
        
        self.action_base = Array(ctypes.c_ubyte, self.size * self.num_actions)
        self.action = np.ctypeslib.as_array(self.action_base.get_obj()).reshape((self.size, self.num_actions))
        
        self.done_base = Array(ctypes.c_ubyte, self.size)
        self.done = np.ctypeslib.as_array(self.done_base.get_obj()).reshape((self.size,))

        self.coll_base = Array(ctypes.c_ubyte, self.size * 2)
        self.coll = np.ctypeslib.as_array(self.coll_base.get_obj()).reshape((self.size, 2))

        self.offroad_base = Array(ctypes.c_ubyte, self.size * 2)
        self.offroad = np.ctypeslib.as_array(self.offroad_base.get_obj()).reshape((self.size, 2))

        self.speed_base = Array(ctypes.c_double, self.size)
        self.speed = np.ctypeslib.as_array(self.speed_base.get_obj()).reshape((self.size, 1))

        self.angle_base = Array(ctypes.c_double, self.size)
        self.angle = np.ctypeslib.as_array(self.angle_base.get_obj()).reshape((self.size, 1))

        self.pos_base = Array(ctypes.c_double, self.size)
        self.pos = np.ctypeslib.as_array(self.pos_base.get_obj()).reshape((self.size, 1))

    def store_data(self, obs, action, done, coll_flag, offroad_flag, speed, angle, trackPos):
        self.obs[self.next_idx.value] = obs
        self.action[self.next_idx.value, action] = 1
        self.done[self.next_idx.value] = done
        self.coll[self.next_idx.value, coll_flag] = 1
        self.offroad[self.next_idx.value, offroad_flag] = 1
        self.speed[self.next_idx.value, 0] = speed
        self.angle[self.next_idx.value, 0] = angle
        self.pos[self.next_idx.value, 0] = trackPos

        self.next_idx.value = (self.next_idx.value + 1) % self.size
        self.num_in_buffer.value = min(self.size, self.num_in_buffer.value + 1)

    def can_sample(self, batch_size):
        return batch_size * self.pred_step + 1 <= self.num_in_buffer.value

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = np.random.choice(self.num_in_buffer - self.pred_step, batch_size)
        return self._encode_sample(idxes), idxes

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([np.concatenate([self._encode_observation(idx+ii)[np.newaxis,:] for ii in range(self.pred_step)], 0)[np.newaxis,:] for idx in idxes], 0)
        nx_obs_batch = np.concatenate([np.concatenate([self._encode_observation(idx+1+ii)[np.newaxis,:] for ii in range(self.pred_step)], 0)[np.newaxis,:] for idx in idxes], 0)
        act_batch = np.concatenate([np.concatenate([self.action[idx+ii, :][np.newaxis,:] for ii in range(self.pred_step)],0)[np.newaxis,:] for idx in idxes], 0)
        sp_batch = np.concatenate([np.concatenate([self.speed[idx+ii,:][np.newaxis,:] for ii in range(self.pred_step+1)],0)[np.newaxis,:] for idx in idxes], 0)
        off_batch = np.concatenate([np.concatenate([self.offroad[idx+ii,:][np.newaxis,:] for ii in range(self.pred_step)],0)[np.newaxis,:] for idx in idxes], 0)
        coll_batch = np.concatenate([np.concatenate([self.coll[idx+ii,:][np.newaxis,:] for ii in range(self.pred_step)], 0)[np.newaxis,:] for idx in idxes], 0)
        pos_batch = np.concatenate([np.concatenate([self.pos[idx+ii,:][np.newaxis,:] for ii in range(self.pred_step+1)],0)[np.newaxis,:] for idx in idxes], 0)
        angle_batch = np.concatenate([np.concatenate([self.angle[idx+ii,:][np.newaxis,:] for ii in range(self.pred_step+1)],0)[np.newaxis,:] for idx in idxes], 0)
        dist_batch = sp_batch*(np.cos(angle_batch)-np.abs(np.sin(angle_batch))-((np.abs(pos_batch))/7.0)**1.0) 

        return act_batch, coll_batch,sp_batch,off_batch,dist_batch,obs_batch, nx_obs_batch, pos_batch 

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)
