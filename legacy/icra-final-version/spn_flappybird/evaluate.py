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
from utils import load_model, ObsBuffer, ActionBuffer, sample_cont_action, train_model, sample_discrete_action
from dqn_agent import DQNAgent
import pickle as pkl
import cv2


def frame(args, obs, video):
    if 'carla' in args.env:
        obs = obs[0]
    # obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    video.write(obs)


def action_log(action):
    with open('action_log.txt', 'a') as f:
        f.write('%0.4f\n' % action[0])


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


def init_models(args):
    train_net = ConvLSTMMulti(args)
    for param in train_net.parameters():
        param.requires_grad = True
    train_net.train()

    net = ConvLSTMMulti(args)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    train_net, epoch = load_model(args.save_path, train_net, data_parallel=args.data_parallel, resume=args.resume)
    net.load_state_dict(train_net.state_dict())

    if torch.cuda.is_available():
        train_net = train_net.cuda()
        net = net.cuda()
        if args.data_parallel:
            train_net = torch.nn.DataParallel(train_net)
            net = torch.nn.DataParallel(net)
    optimizer = optim.Adam(train_net.parameters(), lr=args.lr, amsgrad=True)

    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
    )

    if args.use_dqn:
        dqn_agent = DQNAgent(args, exploration, args.save_path)
        if args.resume:
            dqn_agent.load_model()
    else:
        dqn_agent = None

    if args.resume:
        try:
            num_imgs_start = max(int(open(args.save_path + '/log_train_torcs.txt').readlines()[-1].split(' ')[1]) - 1000,0)
        except:
            print('cannot find file, num_imgs_start is 0')
            num_imgs_start = 0
    else:
        num_imgs_start = 0

    return train_net, net, optimizer, epoch, exploration, dqn_agent, num_imgs_start


class BufferManager:
    def __init__(self, args=None):
        self.args = args
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

    def step_first(self, obs, info):
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
        self.rewards += reward

        return self.mpc_ret, obs_var

    def store_effect(self, action, reward, done):
        self.prev_act = copy.deepcopy(action)
        act_var = Variable(torch.from_numpy(self.action_buffer.store_frame(action)), requires_grad=False) if self.args.lstm2 else None
        if done:
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


class ActionSampleManager:
    def __init__(self, args):
        self.args = args
        self.prev_act = np.array([0.0]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def sample_action(self, net, dqn_net, obs, obs_var, action_var, exploration, tt, avg_img, std_img, info, no_explore=False, must_explore=False):
        action = sample_cont_action(self.args, net, obs_var, info=info, prev_action=np.array([0.0]), avg_img=avg_img, std_img=std_img, tt=tt, action_var=action_var)
        return action, None

    def reset(self):
        self.prev_act = np.array([0.0]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def process_act(self, act, dqn_act):
        return act, dqn_act


def evaluate_policy(args, env, num_steps=40000000):
    ''' create model '''
    net = init_model(args)

    ''' load buffers '''
    buffer_manager = BufferManager(args)
    action_manager = ActionSampleManager(args)

    if args.recording:
        video = cv2.VideoWriter("fb_policy_%d.avi" % args.seed, cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (288, 512), True)

    obs, info = env.reset()
    print(obs.shape)
    if args.recording:
        frame(args, env.real_obs, video)

    obs, reward, done, info = env.step(buffer_manager.prev_act)
    if args.recording:
        frame(args, env.real_obs, video)

    seg = info['segmentation']
    buffer_manager.step_first(obs, info)
    action_var = Variable(torch.from_numpy(np.array([0.0])).repeat(1, args.frame_history_len - 1, 1), requires_grad=False)
    
    if args.recording:
        if os.path.exists('action_log.txt'):
            os.remove('action_log.txt')
        for i in range(4000):
            ret, obs_var = buffer_manager.store_frame(obs, reward, seg)
            if args.normalize:
                avg_img, std_img = buffer_manager.img_buffer.get_avg_std()
            else:
                avg_img, std_img = None, None
            action, dqn_action = action_manager.sample_action(net, None, obs, obs_var, action_var, None, None, avg_img, std_img, info, no_explore=True, must_explore=False)

            obs, reward, done, info = env.step(action)
            seg = info['segmentation']
            print('Action = %d, Reward = %d, Done = %s' % (int(action), reward, str(done)))
            action_log(action)
            frame(args, env.real_obs, video)
            if done:
                break
            action_var = buffer_manager.store_effect(action, reward, done)

        print('\033[1;32mEpisode Finished!\033[0m')
        video.release()