import os
import copy
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from memory import SPCBuffer
from utils import setup_logger, sample_action, get_guide_action


class ObsBuffer:
    def __init__(self, frame_history_len=3):
        self.frame_history_len = frame_history_len
        self.last_obs_all = []

    def store_frame(self, frame):
        obs_np = frame.transpose(2, 0, 1)
        if len(self.last_obs_all) < self.frame_history_len:
            self.last_obs_all = []
            for ii in range(self.frame_history_len):
                self.last_obs_all.append(obs_np)
        else:
            self.last_obs_all = self.last_obs_all[1:] + [obs_np]
        return np.concatenate(self.last_obs_all, 0)

    def clear(self):
        self.last_obs_all = []
        return


class ActionBuffer:
    def __init__(self, frame_history_len=3):
        self.frame_history_len = frame_history_len
        self.last_action_all = []

    def store_frame(self, action):
        action = action.reshape(1, -1)
        if len(self.last_action_all) < self.frame_history_len:
            self.last_action_all = []
            for ii in range(self.frame_history_len):
                self.last_action_all.append(action)
        else:
            self.last_action_all = self.last_action_all[1:] + [action]
        return np.concatenate(self.last_action_all, 0)[np.newaxis, ]

    def clear(self):
        self.last_action_all = []
        return


class BufferManager:
    def __init__(self, args=None):
        self.args = args
        mode = 'eval' if args.eval else 'train'
        self.logger = setup_logger(mode, os.path.join(args.save_path, 'log_{}_{}.txt'.format(mode, args.env)))

        self.spc_buffer = SPCBuffer(args)
        if args.resume:
            self.spc_buffer.load(args.save_path)
        self.obs_buffer = ObsBuffer(args.frame_history_len)
        self.action_buffer = ActionBuffer(args.frame_history_len - 1)
        self.rewards = 0.0
        self.prev_act = np.array([1.0, 0.0])

        self.reward = 0.0
        self.collision_buffer = []
        self.offroad_buffer = []
        self.idx_buffer = []
        self.dist_sum = 0.0

    def store_frame(self, obs, info):
        past_n_frames = self.obs_buffer.store_frame(obs)
        obs_var = Variable(torch.from_numpy(past_n_frames).unsqueeze(0).float().cuda())

        self.spc_buffer.store_frame(obs=obs,
                                    collision=info['collision'],
                                    offroad=info['offroad'],
                                    speed=info['speed'],
                                    seg=info['seg'])
        self.idx_buffer.append(self.spc_buffer.last_idx)
        self.dist_sum += info['speed']
        return obs_var

    def store_effect(self, guide_action, action, reward, done, collision, offroad):
        self.collision_buffer.append(collision)
        self.offroad_buffer.append(offroad)
        self.prev_act = copy.deepcopy(action)
        act_var = Variable(torch.from_numpy(self.action_buffer.store_frame(action)), requires_grad=False).float()
        self.spc_buffer.store_action(guide_action, action, done)
        self.reward += reward
        return act_var

    def reset(self, step):
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.prev_act = np.array([1.0, 0.0])

        self.logger.info('step {} reward {}'.format(step, self.reward))

        # construct labels for self-imitation learning
        epi_len = len(self.idx_buffer)
        idx_buffer = np.array(self.idx_buffer)
        collision_buffer = np.array(self.collision_buffer)
        collision_buffer = np.array([np.sum(collision_buffer[i:i + self.args.safe_length_collision]) == 0 for i in range(collision_buffer.shape[0])])
        offroad_buffer = np.array(self.offroad_buffer)
        offroad_buffer = np.array([np.sum(offroad_buffer[i:i + self.args.safe_length_offroad]) == 0 for i in range(offroad_buffer.shape[0])])
        safe_buffer = collision_buffer * offroad_buffer * self.dist_sum
        self.spc_buffer.expert[idx_buffer] = safe_buffer
        self.spc_buffer.epi_lens.append(epi_len)

        self.idx_buffer = []
        self.collision_buffer = []
        self.offroad_buffer = []
        self.dist_sum = 0.0
        self.reward = 0.0

    def save_spc_buffer(self):
        return # Saving an object larger than 4 GiB causes overflow error
        self.spc_buffer.save(self.args.save_path)

    def load_spc_buffer(self):
        self.spc_buffer.load(self.args.save_path)


class ActionSampleManager:
    def __init__(self, args, guides):
        self.args = args
        self.prev_act = np.array([1.0, 0.0])
        self.guides = guides
        self.p = None

    def sample_action(self, net, obs, obs_var, action_var, exploration, step, explore=False, testing=False):
        if random.random() <= 1 - exploration.value(step) or not explore:
            if self.args.use_guidance:  # sample action distribution p
                obs = Variable(torch.from_numpy(np.expand_dims(obs.transpose(2, 0, 1), axis=0)).float()) / 255.0
                if torch.cuda.is_available():
                    obs = obs.cuda()
                with torch.no_grad():
                    obs = obs.repeat(max(1, torch.cuda.device_count()), 1, 1, 1)
                    self.p = net(obs, function='guide_action')[0]
                    p = F.softmax(self.p / self.args.temperature, dim=-1).data.cpu().numpy()
            else:
                p = None
            action = sample_action(self.args, p, net, obs_var, self.guides, action_var=action_var, testing=testing)
        else:
            action = np.random.rand(self.args.num_total_act) * 2 - 1
        action = np.clip(action, -1, 1)
        guide_act = get_guide_action(self.args.bin_divide, action)
        self.prev_act = action
        return action, guide_act

    def reset(self):
        self.prev_act = np.array([1.0, 0.0])
