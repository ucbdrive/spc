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
        self.coll = None
        self.offroad = None
        self.pos = None
        self.angle = None
        self.speed = None
        self.seg = None
        self.xyz = None
        self.otherlane = None

    def sample_seg(self, batch_size):
        idxes = np.random.choice(range(self.num_in_buffer), batch_size)
        obs = Variable(torch.from_numpy(np.concatenate([self.obs[idx][np.newaxis,:] for idx in idxes], 0)).float() / 255.0, requires_grad=False)
        seg = Variable(torch.from_numpy(np.concatenate([self.seg[idx] for idx in idxes], 0)).long(), requires_grad=False)
        if torch.cuda.is_available():
            obs = obs.cuda()
            seg = seg.cuda()
        return obs, seg

    def sample_collision(self, batch_size):
        idxes_1, _ = np.where(self.coll[:self.num_in_buffer] > 0)
        idxes_2, _ = np.where(self.coll[:self.num_in_buffer] == 0)
        idxes = list(np.random.choice(list(idxes_1), int(batch_size/2))) + list(np.random.choice(list(idxes_2), int(batch_size/2)))
        if self.args.one_hot:
            feature = Variable(torch.from_numpy(np.concatenate([OneHotEncoder(n_values=self.args.classes, sparse=False).fit_transform(self.seg[idx, 0]).reshape(256, 256, 1, self.args.classes).transpose(2, 3, 0, 1) for idx in idxes], 0)).float(), requires_grad=False)
        else:
            feature = Variable(torch.from_numpy(np.concatenate([self.seg[idx] for idx in idxes], 0)).float(), requires_grad=False)
        collision = Variable(torch.cat([torch.ones(int(batch_size/2)), torch.zeros(int(batch_size/2))], dim=-1).long(), requires_grad=False)
        if torch.cuda.is_available():
            feature = feature.cuda()
            collision = collision.cuda()
        return feature, collision

    def sample_offroad(self, batch_size):
        idxes_1, _ = np.where(self.offroad[:self.num_in_buffer] > 0)
        idxes_2, _ = np.where(self.offroad[:self.num_in_buffer] == 0)
        idxes = list(np.random.choice(list(idxes_1), int(batch_size/2))) + list(np.random.choice(list(idxes_2), int(batch_size/2)))
        if self.args.one_hot:
            feature = Variable(torch.from_numpy(np.concatenate([OneHotEncoder(n_values=self.args.classes, sparse=False).fit_transform(self.seg[idx, 0]).reshape(256, 256, 1, self.args.classes).transpose(2, 3, 0, 1) for idx in idxes], 0)).float(), requires_grad=False)
        else:
            feature = Variable(torch.from_numpy(np.concatenate([self.seg[idx] for idx in idxes], 0)).float(), requires_grad=False)
        offroad = Variable(torch.cat([torch.ones(int(batch_size/2)), torch.zeros(int(batch_size/2))], dim=-1).long(), requires_grad=False)
        if torch.cuda.is_available():
            feature = feature.cuda()
            offroad = offroad.cuda()
        return feature, offroad

    def sample_distance(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            idx = random.randint(self.args.frame_history_len, self.num_in_buffer - self.args.pred_step)
            if idx not in idxes and np.sum(self.done[idx-self.args.frame_history_len+1:idx+1]) == 0:
                idxes.append(idx)
        if self.args.one_hot:
            feature = Variable(torch.from_numpy(np.concatenate([OneHotEncoder(n_values=self.args.classes, sparse=False).fit_transform(self.seg[(idx-self.args.frame_history_len+1):(idx+1), 0].reshape(self.args.frame_history_len, 256*256)).reshape(self.args.frame_history_len, 256, 256, 1, self.args.classes).transpose(3, 0, 4, 1, 2).reshape(1, self.args.frame_history_len*self.args.classes, 256, 256) for idx in idxes], 0)).float(), requires_grad=False)
        else:
            feature = Variable(torch.from_numpy(np.concatenate([self.seg[(idx-self.args.frame_history_len+1):(idx+1)].reshape(1, self.args.frame_history_len, 256, 256) for idx in idxes], 0)).float(), requires_grad=False)
        idxes = np.array(idxes)
        distance = Variable(torch.from_numpy(self.speed[idxes] * (np.cos(self.angle[idxes]) - np.abs(np.sin(self.angle[idxes])))).float(), requires_grad=False)
        if torch.cuda.is_available():
            feature = feature.cuda()
            distance = distance.cuda()
        return feature, distance

    def sample_seq(self):
        idx = random.randint(0, self.num_in_buffer - 2)
        while np.sum(self.done[idx-self.args.frame_history_len+1:idx+21]) > 0:
            idx = random.randint(0, self.num_in_buffer - 2)
        feature = Variable(torch.from_numpy(np.concatenate([OneHotEncoder(n_values=self.args.classes, sparse=False).fit_transform(self.seg[idx+i, 0]).reshape(256, 256, 1, self.args.classes).transpose(2, 3, 0, 1) for i in range(1-self.args.frame_history_len, 1)], 0).reshape(1, self.args.frame_history_len*self.args.classes, 256, 256)).float(), requires_grad=False)
        action = Variable(torch.from_numpy(self.action[idx: idx+20][np.newaxis, :]), requires_grad=False)
        seg = Variable(torch.from_numpy(self.seg[idx+1: idx+21].reshape(20, 256, 256)).long(), requires_grad=False)
        signals = dict()
        signals['collision'] = torch.from_numpy(self.coll[idx+1: idx+21].reshape(20)).long()
        signals['offroad'] = torch.from_numpy(self.offroad[idx+1: idx+21].reshape(20)).long()

        if torch.cuda.is_available():
            feature = feature.cuda()
            action = action.cuda()
            seg = seg.cuda()
        return feature, action, seg, signals

    def sample_fcn(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            idx = random.randint(self.args.frame_history_len, self.num_in_buffer - self.args.pred_step)
            if idx not in idxes and np.sum(self.done[idx-self.args.frame_history_len+1:idx+1]) == 0:
                idxes.append(idx)
        if self.args.one_hot:
            feature = Variable(torch.from_numpy(np.concatenate([OneHotEncoder(n_values=self.args.classes, sparse=False).fit_transform(self.seg[(idx-self.args.frame_history_len+1):(idx+1), 0].reshape(256*self.args.frame_history_len, 256)).reshape(self.args.frame_history_len, 256, 256, 1, self.args.classes).transpose(3, 0, 4, 1, 2).reshape(1, self.args.frame_history_len*self.args.classes, 256, 256) for idx in idxes], 0)).float(), requires_grad=False)
        else:
            feature = Variable(torch.from_numpy(np.concatenate([self.seg[(idx-self.args.frame_history_len+1):(idx+1)].reshape(1, self.args.frame_history_len, 256, 256) for idx in idxes], 0)).float(), requires_grad=False)
        action = Variable(torch.from_numpy(np.concatenate([self.action[idx].reshape(1, 2) for idx in idxes], 0)), requires_grad=False)
        seg = Variable(torch.from_numpy(np.concatenate([self.seg[idx+1] for idx in idxes], 0)).long(), requires_grad=False)
        if torch.cuda.is_available():
            feature = feature.cuda()
            action = action.cuda()
            seg = seg.cuda()
        return feature, action, seg

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

        if self.args.lstm2:
            data_dict['prev_action'] = np.concatenate([self.action[idx-self.args.frame_history_len+1: idx, :][np.newaxis, :] for idx in idxes], 0)
        else:
            data_dict['prev_action'] = None

        if self.args.use_collision:
            data_dict['coll_batch'] = np.concatenate([self.coll[idx + 1: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)

        if self.args.use_offroad:
            data_dict['off_batch'] = np.concatenate([self.offroad[idx + 1: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)

        if self.args.use_otherlane:
            data_dict['otherlane_batch'] = np.concatenate([self.otherlane[idx + 1: idx + self.args.pred_step + 1, :][np.newaxis, :] for idx in idxes], 0)

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
            self.done = np.empty([self.args.buffer_size], dtype=np.int32)

            if self.args.use_collision:
                self.coll = np.empty([self.args.buffer_size, 1], dtype=np.int32)
            if self.args.use_offroad:
                self.offroad = np.empty([self.args.buffer_size, 1], dtype=np.int32)  # if 'torcs' in self.args.env else np.float32)
            if self.args.use_otherlane:
                self.otherlane = np.empty([self.args.buffer_size, 1], dtype=np.float32)

            if self.args.use_pos:
                self.pos = np.empty([self.args.buffer_size, 1], dtype=np.float32)
            self.angle = np.empty([self.args.buffer_size, 1], dtype=np.float32)
            self.speed = np.empty([self.args.buffer_size, 1], dtype=np.float32)

            if self.args.use_seg:
                self.seg = np.empty([self.args.buffer_size] + [1, 256, 256], dtype=np.uint8)

            if self.args.use_xyz:
                self.xyz = np.empty([self.args.buffer_size, 3], dtype=np.float32)

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

    def store_effect(self, idx, coll, off, speed, otherlane=None, angle=None, pos=None, xyz=None, seg=None):
        if self.args.use_xyz:
            self.xyz[idx, :] = xyz

        if self.args.use_seg:
            self.seg[idx, :] = seg

        if self.args.use_collision:
            self.coll[idx, 0] = int(coll)

        if self.args.use_offroad:
            if True:  # 'torcs' in self.args.env:
                off = int(off)
            self.offroad[idx, 0] = off

        if self.args.use_otherlane:
            self.otherlane[idx, 0] = otherlane

        self.speed[idx, 0] = speed

        if self.args.use_angle or self.args.use_distance:
            self.angle[idx, 0] = angle

        if self.args.use_pos:
            self.pos[idx, 0] = pos

    def load(self, path):
        return
        path = os.path.join(path, 'MPCBuffer')
        try:
            assert os.path.isdir(path)

            assert os.path.exists(os.path.join(path, 'obs.npy'))
            assert os.path.exists(os.path.join(path, 'action.npy'))
            assert os.path.exists(os.path.join(path, 'done.npy'))

            if self.args.use_collision:
                assert os.path.exists(os.path.join(path, 'coll.npy'))
            if self.args.use_offroad:
                assert os.path.exists(os.path.join(path, 'offroad.npy'))
            if self.args.use_otherlane:
                assert os.path.exists(os.path.join(path, 'otherlane.npy'))

            if self.args.use_pos:
                assert os.path.exists(os.path.join(path, 'pos.npy'))
            if self.args.use_angle:
                assert os.path.exists(os.path.join(path, 'angle.npy'))
            if self.args.use_speed:
                assert os.path.exists(os.path.join(path, 'speed.npy'))

            if self.args.use_seg:
                assert os.path.exists(os.path.join(path, 'seg.npy'))
            if self.args.use_xyz:
                assert os.path.exists(os.path.join(path, 'xyz.npy'))

            assert os.path.exists(os.path.join(path, 'mpc_buffer.json'))

            self.obs = np.load(os.path.join(path, 'obs.npy'))
            self.action = np.load(os.path.join(path, 'action.npy'))
            self.done = np.load(os.path.join(path, 'done.npy'))

            if self.args.use_collision:
                self.coll = np.load(os.path.join(path, 'coll.npy'))
            if self.args.use_offroad:
                self.offroad = np.load(os.path.join(path, 'offroad.npy'))
            if self.args.use_otherlane:
                self.otherlane = np.load(os.path.join(path, 'otherlane.npy'))

            if self.args.use_pos:
                self.pos = np.load(os.path.join(path, 'pos.npy'))
            if self.args.use_angle:
                self.angle = np.load(os.path.join(path, 'angle.npy'))
            self.speed = np.load(os.path.join(path, 'speed.npy'))

            if self.args.use_seg:
                self.seg = np.load(os.path.join(path, 'seg.npy'))
            if self.args.use_xyz:
                self.xyz = np.load(os.path.join(path, 'xyz.npy'))

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

        if self.args.use_collision:
            np.save(os.path.join(path, 'coll.npy'), self.coll)
        if self.args.use_offroad:
            np.save(os.path.join(path, 'offroad.npy'), self.offroad)
        if self.args.use_otherlane:
            np.save(os.path.join(path, 'otherlane.npy'), self.otherlane)

        if self.args.use_pos:
            np.save(os.path.join(path, 'pos.npy'), self.pos)
        if self.args.use_angle:
            np.save(os.path.join(path, 'angle.npy'), self.angle)
        np.save(os.path.join(path, 'speed.npy'), self.speed)

        if self.args.use_seg:
            np.save(os.path.join(path, 'seg.npy'), self.seg)
        if self.args.use_xyz:
            np.save(os.path.join(path, 'xyz.npy'), self.xyz)

        state_dict = {'next_idx': self.next_idx, 'num_in_buffer': self.num_in_buffer}
        with open(os.path.join(path, 'mpc_buffer.json'), "w") as f:
            json.dump(state_dict, f, indent=4)
