from model import ConvLSTMMulti
import numpy as np
import random
import os
import torch
from torch.autograd import Variable
from dqn_utils import PiecewiseSchedule
from mpc_utils import MPCBuffer, IMGBuffer
import copy
from torch import optim
from utils import load_model, ObsBuffer, ActionBuffer, sample_cont_action, train_model, sample_discrete_action, draw_from_pred, from_variable_to_numpy
from dqn_agent import DQNAgent
import pickle as pkl
import cv2
import pdb

def init_model(args):
    net = ConvLSTMMulti(args)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    state_dict = torch.load('best.pt')
    net.load_state_dict(state_dict)

    if torch.cuda.is_available():
        net = net.cuda()
        if args.data_parallel:
            net = torch.nn.DataParallel(net)

    return net


class BufferManager:
    def __init__(self, args=None):
        self.args = args
        self.mpc_buffer = MPCBuffer(args)
        # if args.resume:
        #     self.mpc_buffer.load(args.save_path)
        self.img_buffer = IMGBuffer()
        self.obs_buffer = ObsBuffer(args.frame_history_len)
        if self.args.lstm2:
            self.action_buffer = ActionBuffer(args.frame_history_len-1)
        self.prev_act = np.array([0.0]) if args.continuous else 1

        self.avg_img = None
        self.std_img = None
        self.epi_rewards = []
        self.rewards = 0.0
        self.mpc_ret = 0

        self._rewards = []
        self.reward_idx = []

    def step_first(self, obs):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()

    def store_frame(self, obs, reward, seg):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
        this_obs_np = self.obs_buffer.store_frame(obs)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0).float().cuda())
        self._rewards.append(reward)
        self.reward_idx.append(self.mpc_ret)
        self.mpc_ret = self.mpc_buffer.store_frame(obs)
        self.rewards += reward
        return self.mpc_ret, obs_var

    def store_effect(self, action, reward, done):
        self.prev_act = copy.deepcopy(action)
        act_var = Variable(torch.from_numpy(self.action_buffer.store_frame(action)), requires_grad=False) if self.args.lstm2 else None
        self.mpc_buffer.store_action(self.mpc_ret, action, done)
        if done:
            for i in range(len(self._rewards)-1, 0, -1):
                self._rewards[i-1] += self.args.gamma * self._rewards[i]
            for i in range(len(self._rewards)):
                self.mpc_buffer.value[self.reward_idx[i], 0] = self._rewards[i]
            self._rewards = []
            self.reward_idx = []
        return act_var

    def update_avg_std_img(self):
        if self.args.normalize:
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()

    def reset(self, step, log_name='log_train_torcs.txt'):
        self.obs_buffer.clear()
        self.epi_rewards.append(self.rewards)
        self.rewards = 0
        self.prev_act = np.array([0.0]) if self.args.continuous else 1
        print('past 100 episode rewards is',
              "{0:.3f}".format(np.mean(self.epi_rewards[-1:])))
        with open(os.path.join(self.args.save_path, log_name), 'a') as fi:
            fi.write('step '+str(step))
            fi.write(' reward ' + str(np.mean(self.epi_rewards[-1:])))
            fi.write(' std ' + str(np.std(self.epi_rewards[-1:])))
            fi.write('\n')

    def save_mpc_buffer(self):
        self.mpc_buffer.save(self.args.save_path)

    def load_mpc_buffer(self):
        self.mpc_buffer.load(self.args.save_path)


class ActionSampleManager:
    def __init__(self, args):
        self.args = args
        self.prev_act = np.array([0.0]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def sample_action(self, net, dqn_net, obs, obs_var, action_var, exploration, tt, avg_img, std_img, info, no_explore=False, must_explore=False, get_first=True):
        if tt % self.args.num_same_step != 0:
            return self.process_act(self.prev_act, self.prev_dqn_act)
        else:
            action = sample_cont_action(self.args, net, obs_var, info=info, prev_action=np.array([0.0]), avg_img=avg_img, std_img=std_img, tt=tt, action_var=action_var, get_first=get_first)
            dqn_act = None
            self.prev_act = action
            self.prev_dqn_act = dqn_act
            return action, dqn_act

    def reset(self):
        self.prev_act = np.array([0.0]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def process_act(self, act, dqn_act):
        return act, dqn_act


def draw(action, _action, step, obs_var, net, args, action_var, name):
    if not os.path.isdir(os.path.join('demo', str(step), name)):
        os.makedirs(os.path.join('demo', str(step), name))
    s = 'Next Action: %d\n' % action
    # cv2.putText(raw_obs, 'Next Action: [%0.1f, %0.1f]' % (action[0], action[1]), (70, 400), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)

    action = _action.view(1, args.pred_step, 1)
    action = Variable(action.cuda().float(), requires_grad=False)
    obs_var = obs_var / 255.0
    obs_var = obs_var.view(1, 1, 9, 256, 128)
    with torch.no_grad():
        output = net(obs_var, action, training=False, action_var=action_var)

    for i in range(args.pred_step):
        img = draw_from_pred(from_variable_to_numpy(torch.argmax(output['seg_pred'][0, i+1], 0)))
        s += 'Step %d\n' % i
        s += 'Action %d\n' % _action[i]
        s += 'Reward: %0.2f\n' % float(output['reward'][0, i, 0])
        s += 'Value: %0.2f\n' % float(output['value'][0, i, 0])
        cv2.imwrite(os.path.join('demo', str(step), name, 'seg%d.png' % (i+1)), img)

    with open(os.path.join('demo', str(step), name, 'log.txt'), 'w') as f:
        f.write(s)


def evaluate_policy(args, env):
    net = init_model(args)

    ''' load buffers '''
    buffer_manager = BufferManager(args)
    action_manager = ActionSampleManager(args)

    cap = cv2.VideoCapture('video.avi')
    ret, raw_obs = cap.read()
    ret, raw_obs = cap.read()
    with open('actions.txt', 'r') as f:
        actions = f.readlines()[1:]
    obs = cv2.resize(cv2.cvtColor(raw_obs, cv2.COLOR_BGR2RGB), (128, 256))
    ret, obs_var = buffer_manager.store_frame(obs, 0, 0)
    buffer_manager.step_first(obs)
    action_var = Variable(torch.from_numpy(np.array([0.0])).repeat(1, args.frame_history_len - 1, 1), requires_grad=False)

    for i in range(1000):
        if args.normalize:
            avg_img, std_img = buffer_manager.img_buffer.get_avg_std()
        else:
            avg_img, std_img = None, None

        _action, dqn_action = action_manager.sample_action(net, None, obs, obs_var, action_var, None, i, avg_img, std_img, None, no_explore=True, must_explore=False, get_first=False)
        action = np.array([int(actions[i])])
        draw(action, _action, i, obs_var, net, args, action_var, 'outcome')
        cv2.imwrite(os.path.join('demo', str(i), 'obs.png'), raw_obs)

        ret, raw_obs = cap.read()
        obs = cv2.resize(cv2.cvtColor(raw_obs, cv2.COLOR_BGR2RGB), (128, 256))
        ret, obs_var = buffer_manager.store_frame(obs, 0, 0)
        action_var = buffer_manager.store_effect(action, 0, 0)

    cap.release()
