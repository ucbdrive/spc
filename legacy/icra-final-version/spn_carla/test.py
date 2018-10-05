import numpy as np
import torch
from torch.autograd import Variable
import copy
from utils import *
from dqn_utils import *
from dqn_agent import *
from mpc_utils import *
from torcs_wrapper import *
import argparse
from args import init_parser
import sys
sys.path.append('/media/xinleipan/data/git/pyTORCS/py_TORCS')
sys.path.append('/media/xinleipan/data/git/pyTORCS/')
from py_TORCS import torcs_envs
from train_cont_policy import *

def test(args, env, net, file_name, dqn_name=None):
    buffer_manager = BufferManager(args)
    action_manager = ActionSampleManager(args)
    done_cnt = 0
    _, info = env.reset()
    obs, reward, done, info = env.step(np.array([1.0, 0.0]))
    buffer_manager.step_first(obs, info)
    exploration = PiecewiseSchedule([(0, 0.0), (1000, 0.0)], outside_value = 0.0)
    if args.use_dqn:
        dqn_agent = DQNAgent(args, exploration, args.save_path)
        dqn_agent.load_model(dqn_name)
    else:
        dqn_agent = None
    while done_cnt < 1:
        seg = env.env.get_segmentation().reshape((1, 256, 256)) if args.use_seg else None
        ret, obs_var = buffer_manager.store_frame(obs, info, seg)
        if args.normalize:
            avg_img, std_img = buffer_manager.img_buffer.get_avg_std()
        else:
            avg_img, std_img = None, None
        action, dqn_action = action_manager.sample_action(net, dqn_agent, obs, obs_var, exploration, 1e8, avg_img, std_img)
        obs, reward, done, info = env.step(action)
        print('action ', "{0:.2f}".format(action[0]), "{0:.2f}".format(action[1]), \
            ' pos ', "{0:.2f}".format(info['trackPos']), \
            "{0:.2f}".format(info['pos'][0]), \
            "{0:.2f}".format(info['pos'][1]),\
            ' angle ', "{0:.2f}".format(info['angle']), \
            ' reward ', "{0:.2f}".format(reward['with_pos']),\
            ' reward without ', "{0:.2f}".format(reward['without_pos'])) 
        buffer_manager.store_effect(action, reward, done)
        buffer_manager.update_avg_std_img()
        if done:
            done_cnt += 1
            obs, prev_info = env.reset()
            obs, _, _, info = env.step(np.array([1.0, 0.0]))
            buffer_manager.reset(prev_info, file_name.split('_')[-1].split('.')[0], log_name='log_test_torcs.txt')
            action_manager.reset()
        if args.use_dqn:
            dqn_agent.store_effect(dqn_action, reward['with_pos'], done)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    envs = torcs_envs(num = 1, game_config = args.game_config, mkey_start = 817 + args.id, screen_id = 160 + args.id,
                      isServer = int(args.xvfb), continuous = args.continuous, resize = True)
    env = envs.get_envs()[0]
    env = TorcsWrapper(env, random_reset=args.use_random_reset, continuous = args.continuous)
    net = ConvLSTMMulti(args)
    net, _ = load_model(args.save_path, net, data_parallel=False, resume=False)
    net.eval()
    tested = []
    while True:
        file_list = os.listdir(os.path.join(args.save_path, 'model'))
        for fi in sorted(file_list):
            file_name = os.path.join(args.save_path, 'model', fi)
            net.load_state_dict(torch.load(file_name), strict=True)
            print('load model', file_name)
            if args.use_dqn:
                dqn_name = 'model_'+str(int(file_name.split('_')[-1].split('.')[0]))+'.pt'
                if not os.path.exists(os.path.join(args.save_path, 'dqn/model', dqn_name)):
                    continue
            if file_name not in tested:
                tested.append(file_name)
                test(args, env, net, file_name, dqn_name) 
