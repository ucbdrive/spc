import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dqn
from dqn_utils import *
import argparse
import torch.multiprocessing as mp
from envs import create_atari_env
import os
from atari_wrappers import *
import test
import sys
import pdb
import time
import dla
from dla import *

parser = argparse.ArgumentParser(description='distributed dqn')
parser.add_argument('--use-cuda', action='store_true', help='use cuda or not')
parser.add_argument('--frame-history-len', type=int, default=4,
                    help='frame history length')
parser.add_argument('--save-path', type=str, default='models', help='model save path')
parser.add_argument('--env-id', type=str, default='torcs-v0', help='environment id')
parser.add_argument('--log-name', type=str, default='log_ours_adv.txt', help='log file name')
parser.add_argument('--load-old-q-value', action='store_true', help='to use old q value or not')
parser.add_argument('--buffer-size', type=int, default=1000000, help='buffer size')
parser.add_argument('--dueling', action='store_true', help='use dueling network or not')
parser.add_argument('--config', type=str, default='quickrace_discrete_single.xml')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-starts', type=int, default=1000)
parser.add_argument('--learning-freq', type=int, default=4)
parser.add_argument('--target-update-freq', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--grad-norm-clipping', type=float, default=10.)
parser.add_argument('--test-with-random-adv', action='store_true')
parser.add_argument('--test-without-adv', action='store_true')
parser.add_argument('--test-adv-freq', type=int, default=5)
parser.add_argument('--train-dqn', action='store_true')
parser.add_argument('--learning-rate', type=float, default=0.00025)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--epsilon-frames', type=int, default=1000000)
parser.add_argument('--save-img', action='store_true')
parser.add_argument('--with-adv', action='store_true', help='SGD attack')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--n-iter', type=int, default=10)
parser.add_argument('--test-with-adv-agent', action='store_true')
parser.add_argument('--use-reset', action='store_true')
parser.add_argument('--without-dla', action='store_true')
parser.add_argument('--early-stop', action='store_true')
parser.add_argument('--num-total-act', type=int, default=6)

class atari_model(nn.Module):
    def __init__(self, in_channels=12, num_actions=18, frame_history_len=4, without_dla=True):
        super(atari_model, self).__init__()
        self.without_dla = without_dla
        if self.without_dla:
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc4 = nn.Linear(7 * 7 * 64, 512)
        else:
            self.dla = dla.dla46x_c(pretrained=True).cuda()
            self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.num_actions = num_actions
        self.frame_history_len = frame_history_len

    def get_feature(self, x):
        res = []
        for i in range(self.frame_history_len):
            res.append(self.dla(x[:,i*3:(i+1)*3,:,:]))
        res = torch.cat(res, dim=1)
        return res

    def forward(self, x):
        if self.without_dla:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        else:
            x = self.get_feature(x)
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        res = self.fc5(x)
        return res

class atari_model_dueling(nn.Module):
    def __init__(self, in_channels=12, num_actions=18):
        super(atari_model_dueling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.num_actions = num_actions
        self.fc6 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        value = self.fc6(x)
        adv   = self.fc5(x)
        value = value.repeat(x.size(0), self.num_actions)
        advavg= torch.mean(adv, dim=1)
        advavg= advavg.repeat(x.size(0), self.num_actions)
        res   = value + adv - advavg
        return res

class atari_model_block(nn.Module):
    def __init__(self, in_channels=12, num_actions=18, num_process=10):
        super(atari_model_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32 ,64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512 * num_process)
        self.fc5 = nn.ModuleList([nn.Linear(512, num_actions) for i in range(num_process)])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        outs = []
        for i, l in enumerate(self.fc5):
            outs.append(self.fc5[i](x[:, i*512:(i+1)*512]))
        outs = torch.cat(outs, 1)
        return outs   

def atari_learn(args,
                env,                
                num_timesteps,
                global_step,
                global_lock):
    num_iterations = float(num_timesteps) / 4.0
    LEARNING_RATE = 5e-5
    lr_multiplier = 3.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    if args.optimizer == 'adam':
        optimizer = dqn.OptimizerSpec(
            constructor=optim.Adam,
            kwargs=dict(lr=args.learning_rate, eps=1e-4)
        )
    elif args.optimizer == 'rmsprop':
        optimizer = dqn.OptimizerSpec(
            constructor=optim.RMSprop,
            kwargs=dict(lr=args.learning_rate, eps=1e-4, momentum=0.5)
        )
    else:
        sys.exit('unspecified optimizer')

    def stopping_criterion(env, t):
        return t > 40000000

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (args.epsilon_frames, 0.02),
            (num_iterations / 2, 0.02),
        ], outside_value=0.02
    )

    if args.dueling == False:
        dqn.learn(
            args,
            env,
            q_func=atari_model,
            optimizer_spec=optimizer,
            exploration=exploration_schedule,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_starts=args.learning_starts,
            learning_freq=args.learning_freq,
            frame_history_len=args.frame_history_len,
            target_update_freq=args.target_update_freq,
            grad_norm_clipping=args.grad_norm_clipping,
            use_cuda=args.use_cuda,
            global_step=global_step,
            global_lock=global_lock,
            load_old_q_value=args.load_old_q_value,
            lr_schedule=lr_schedule
        )
    else:
        dqn.learn(
            args,
            env,
            q_func=atari_model_dueling,
            optimizer_spec=optimizer,
            exploration=exploration_schedule,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_starts=args.learning_starts,
            learning_freq=args.learning_freq,
            frame_history_len=args.frame_history_len,
            target_update_freq=args.target_update_freq,
            grad_norm_clipping=args.grad_norm_clipping,
            use_cuda=args.use_cuda,
            global_step=global_step,
            global_lock=global_lock,
            load_old_q_value=args.load_old_q_value
        )
    env.close()

def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_env(env_id, seed):
    env = gym.make(env_id)
    env.seed(seed)
    env = wrap_deepmind(env)
    return env

if __name__ == "__main__":
    args = parser.parse_args()
    args.save_path = 'results/'+args.save_path   
    if args.env_id == 'torcs-v0' and args.config == 'quickrace_discrete_multi.xml':
        env = create_atari_env(args.env_id, reward_ben=True, config=args.config, rescale=False)
    elif args.env_id == 'torcs-v0' and args.config == 'quickrace_discrete_single.xml' and args.save_img == True:
        env = create_atari_env(args.env_id, reward_ben=True, config=args.config, rescale=False)
    elif args.env_id == 'torcs-v0' and args.config == 'quickrace_discrete_single.xml' and args.save_img == False:
        env = create_atari_env(args.env_id, reward_ben=True, config=args.config, rescale=True)
    elif args.env_id == 'torcs-v0':
        env = create_atari_env(args.env_id, reward_ben=True, config=args.config, rescale=True)
    else:
        env = get_env(args.env_id, args.seed)
    set_global_seeds(args.seed)
    if args.train_dqn == True:    
        global_step = mp.Value('i', 0)
        global_lock = mp.Lock()
        if os.path.isdir('results') == False:
            os.mkdir('results')
        if os.path.isdir(args.save_path) == False:
            os.mkdir(args.save_path)
        if os.path.isdir(args.save_path+'/models') == False:
            os.mkdir(args.save_path+'/models')
        atari_learn(args, env, 400000000, global_step, global_lock)

    elif args.test_without_adv == True or args.test_with_random_adv == True or args.test_with_adv_agent == True:
        tested = []
        # enable testing from pretested
        try:
            past_logs = open(os.path.join(args.save_path, args.log_name)).readlines()
        except:
            past_logs = []
        model_paths = os.listdir(args.save_path+'/models')
        model_paths_new = [int(model_paths[i]) for i in range(len(model_paths))]
        model_paths_new = sorted(model_paths_new)
        for i in range(len(past_logs)):
            load_path = os.path.join(args.save_path, 'models', str(model_paths_new[i]))
            if load_path not in tested:
                tested.append(load_path)
        while True:
            model_paths = os.listdir(args.save_path+'/models')
            model_paths_new = [int(model_paths[i]) for i in range(len(model_paths))]
            model_paths_new = sorted(model_paths_new)
            adv_path_list = '/media/xinleipan/data/RARARL/discrete_pt/bootstrap_dqn_shared_adv/results/'+\
                'bootboot_torcs_nproc_10_buffer_100k_batch_32_lr_1e4_poi_0p5_4_1000_500k_withvar_withadv_advfreq_5_adam_block_risk_averse_poisson_reset_lamda_0p1_new_0301/models'
            adv_models = os.listdir(adv_path_list)
            adv_models_new = sorted([int(adv_models[i]) for i in range(len(adv_models))])
            adv_path = os.path.join(adv_path_list, str(adv_models_new[-1]))
            for i in range(len(model_paths_new)):
                load_path = os.path.join(args.save_path, 'models', str(model_paths_new[i]))
                num = int(model_paths_new[i]/1000)*1000
                if load_path not in tested and num % 30000 == 0:# and num > 5310032:
                    tested.append(load_path)
                    if args.dueling == True:
                        test.test(args, env, atari_model_dueling, frame_hisotry_len=args.frame_history_len, load_path = load_path, adv_path=adv_path, adv_model=atari_model_block)
                    else:
                        test.test(args, env, atari_model, frame_history_len = args.frame_history_len, load_path = load_path, adv_path=adv_path, adv_model=atari_model_block)
