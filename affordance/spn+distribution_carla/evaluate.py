from model import ConvLSTMMulti
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
from dqn_utils import *
import copy
import cv2
from utils import *
from torcs_wrapper import TorcsWrapper
from dqn_agent import *
import pickle as pkl
import pdb


def frame(args, obs, video):
    if 'carla' in args.env:
        obs = obs[0]
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    video.write(obs)


def action_log(action):
    with open('action_log.txt', 'a') as f:
        f.write('[%0.4f, %0.4f]\n' % (action[0], action[1]))


def init_model(args):
    net = ConvLSTMMulti(args)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    state_dict = torch.load('trained_model.pth')
    net.load_state_dict(state_dict)

    if torch.cuda.is_available():
        net = net.cuda()
        if args.data_parallel:
            net = torch.nn.DataParallel(net)

    return net


class BufferManager:
    def __init__(self, args=None):
        self.args = args
        if self.args.normalize:
            self.img_buffer = IMGBuffer()
        self.obs_buffer = ObsBuffer(args.frame_history_len)
        if self.args.lstm2:
            self.action_buffer = ActionBuffer(args.frame_history_len-1)
        self.epi_rewards = []
        self.rewards = 0.0
        self.prev_act = np.array([1.0, -0.1]) if args.continuous else 1

        self.avg_img = None
        self.std_img = None
        self.speed_np = None
        self.pos_np = None
        self.posxyz_np = None
        self.prev_xyz = None
        self.epi_rewards_with = []
        self.epi_rewards_without = []
        self.rewards_with = 0.0
        self.rewards_without = 0.0
        self.mpc_ret = 0

    def step_first(self, obs, info):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()
        if 'torcs' in self.args.env:
            self.speed_np, self.pos_np, self.posxyz_np = get_info_np(info, use_pos_class=False)
            self.prev_xyz = np.array(info['pos'])


    def store_frame(self, obs):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
        this_obs_np = self.obs_buffer.store_frame(obs)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0).float().cuda())

        return obs_var

    def store_effect(self, action, reward):
        self.prev_act = copy.deepcopy(action)
        act_var = Variable(torch.from_numpy(self.action_buffer.store_frame(action)), requires_grad=False) if self.args.lstm2 else None
        self.rewards_with += reward['with_pos']
        self.rewards_without += reward['without_pos']
        return act_var

    def update_avg_std_img(self):
        if self.args.normalize:
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()

    def reset(self, info, step, log_name='log_evaluate_torcs.txt'):
        self.obs_buffer.clear()
        self.epi_rewards_with.append(self.rewards_with)
        self.epi_rewards_without.append(self.rewards_without)
        self.rewards_with, self.rewards_without = 0, 0
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1
        if 'torcs' in self.args.env:
            self.speed_np, self.pos_np, self.posxyz_np = get_info_np(info, use_pos_class=False)
            self.prev_xyz = np.array(info['pos'])
        print('past 100 episode rewards is',
              "{0:.3f}".format(np.mean(self.epi_rewards_with[-1:])),
              ' without is ', "{0:.3f}".format(np.mean(self.epi_rewards_without[-1:])))
        with open(self.args.save_path+'/'+log_name, 'a') as fi:
            fi.write('step '+str(step))
            fi.write(' reward_with ' + str(np.mean(self.epi_rewards_with[-1:])))
            fi.write(' std ' + str(np.std(self.epi_rewards_with[-1:])))
            fi.write(' reward_without ' + str(np.mean(self.epi_rewards_without[-1:])))
            fi.write(' std ' + str(np.std(self.epi_rewards_without[-1:])) + '\n')

    def save_mpc_buffer(self):
        self.mpc_buffer.save(self.args.save_path)

    def load_mpc_buffer(self):
        self.mpc_buffer.load(self.args.save_path)


class ActionSampleManager:
    def __init__(self, args):
        self.args = args
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def sample_action(self, net, dqn_net, obs, obs_var, action_var, avg_img, std_img, info):
        if False:
            return self.process_act(self.prev_act, self.prev_dqn_act)
        else:
            dqn_act = None
            if self.args.continuous:
                action = sample_cont_action(self.args, net, obs_var, info=info, prev_action=np.array([0.5, 0.01]), avg_img=avg_img, std_img=std_img, tt=0, action_var=action_var)
                action = np.clip(action, -1, 1)
                if self.args.use_dqn:
                    dqn_act = dqn_net.sample_action(obs, tt)
            else:
                with torch.no_grad():
                    action = sample_discrete_action(self.args, net, obs_var, prev_action=self.prev_act)
            action, dqn_act = self.process_act(action, dqn_act)
            self.prev_act = action
            self.prev_dqn_act = dqn_act
            return action, dqn_act

    def reset(self):
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def process_act(self, act, dqn_act):
        if self.args.use_dqn and self.args.continuous:
            if abs(act[1]) <= dqn_act * 0.1:
                act[1] = 0
        elif self.args.continuous and not self.args.use_dqn:
            if abs(act[1]) <= 0.0:
                act[1] = 0
        return act, dqn_act


def draw(action, step, obs_var, net, args, action_var, name):
    if not os.path.isdir(os.path.join('demo', str(step), name)):
        os.makedirs(os.path.join('demo', str(step), name))
    s = 'Next Action: [%0.1f, %0.1f]\n' % (action[0], action[1])
    # cv2.putText(raw_obs, 'Next Action: [%0.1f, %0.1f]' % (action[0], action[1]), (70, 400), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)

    action = torch.from_numpy(action).view(1, 1, 2).repeat(1, args.pred_step, 1)
    action = Variable(action.cuda().float(), requires_grad=False)
    obs_var = obs_var / 255.0
    obs_var = obs_var.view(1, 1, 9, 256, 256)
    with torch.no_grad():
        output = net(obs_var, action, training=False, action_var=action_var)
        output['offroad_prob'] = F.softmax(output['offroad_prob'], -1)
        output['coll_prob'] = F.softmax(output['coll_prob'], -1)

    for i in range(args.pred_step):
        img = draw_from_pred_torcs(from_variable_to_numpy(torch.argmax(output['seg_pred'][0, i+1], 0)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.putText(img, 'OffRoad: %0.2f%%' % float(100*output['offroad_prob'][0, i, 1]), (20, 160), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
        # cv2.putText(img, 'Collision: %0.2f%%' % float(100*output['coll_prob'][0, i, 1]), (20, 200), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
        s += 'Step %d\n' % i
        s += 'OffRoad: %0.2f%%\n' % float(100*output['offroad_prob'][0, i, 1])
        s += 'Collision: %0.2f%%\n' % float(100*output['coll_prob'][0, i, 1])
        s += 'Distance: %0.2f%%\n' % float(output['dist'][0, i, 0])
        cv2.imwrite(os.path.join('demo', str(step), name, 'seg%d.png' % (i+1)), img)

    with open(os.path.join('demo', str(step), name, 'log.txt'), 'w') as f:
        f.write(s)


def evaluate_policy(args, env):
    ''' basics '''
    if 'torcs' in args.env:
        env = TorcsWrapper(env, random_reset=args.use_random_reset, continuous=args.continuous, imsize=(640, 480))

    ''' create model '''
    net = init_model(args)

    ''' load buffers '''
    buffer_manager = BufferManager(args)
    action_manager = ActionSampleManager(args)

    if args.recording:
        video = cv2.VideoWriter("torcs_policy.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (640, 480), True)

    obs, info = env.reset()
    print(obs.shape)
    if args.recording:
        frame(args, obs, video)
    obs, reward, done, info = env.step(buffer_manager.prev_act)
    raw_obs = obs
    if args.recording:
        frame(args, obs, video)
    if 'carla' in args.env:
        obs, seg = obs
    obs = cv2.resize(obs, (256, 256))
    obs_var = buffer_manager.store_frame(obs)
    buffer_manager.step_first(obs, info)
    action_var = Variable(torch.from_numpy(np.array([-1.0, 0.0])).repeat(1, args.frame_history_len - 1, 1), requires_grad=False)

    if args.visualize:
        with open('action_log.txt', 'r') as f:
            actions = list(map(lambda x: np.array(eval(x)), f.readlines()))

        stops = range(1000)
        steps = max(stops) + 1

        for i in range(steps):
            if args.normalize:
                avg_img, std_img = buffer_manager.img_buffer.get_avg_std()
            else:
                avg_img, std_img = None, None

            action = actions[i]
            if True:  # i in stops:
                draw(action, i, obs_var, net, args, action_var, 'outcome')
                cv2.imwrite(os.path.join('demo', str(i), 'obs.png'), cv2.cvtColor(raw_obs, cv2.COLOR_BGR2RGB))

            obs, reward, done, info = env.step(action)
            # action_log(action)
            if done:
                break
            if 'carla' in args.env:
                obs, seg = obs
            raw_obs = obs
            obs = cv2.resize(obs, (256, 256))
            obs_var = buffer_manager.store_frame(obs)
            action_var = buffer_manager.store_effect(action, reward)

        # action = actions[steps]

    if args.recording:
        if os.path.exists('action_log.txt'):
            os.remove('action_log.txt')
        for i in range(4000):
            obs_var = buffer_manager.store_frame(obs)
            if args.normalize:
                avg_img, std_img = buffer_manager.img_buffer.get_avg_std()
            else:
                avg_img, std_img = None, None
            action, dqn_action = action_manager.sample_action(net, None, obs, obs_var, action_var, avg_img, std_img, info)
            obs, reward, done, info = env.step(action)
            action_log(action)
            frame(args, obs, video)
            if done:
                break
            if 'carla' in args.env:
                obs, seg = obs
            obs = cv2.resize(obs, (256, 256))
            action_var = buffer_manager.store_effect(action, reward)
        print('\033[1;32mEpisode Finished!\033[0m')
        video.release()