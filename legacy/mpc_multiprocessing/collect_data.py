from __future__ import division, print_function
import os
import time
import cv2
import copy
import pickle as pkl
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from envs import create_env
from model import ConvLSTMMulti
from utils import *
from dqn_utils import PiecewiseSchedule

def resize_and_normalize(obs, args, avg_img = 0, std_img = 1):
    obs = cv2.resize(obs, (args.observation_height, args.observation_width))
    obs = (obs - avg_img) / (std_img + 0.0001) if args.normalize else obs / 255.0
    return obs.transpose(2, 0, 1)

def collect_data(args, memory_pool):
    logger = setup_logger('collector', os.path.join(args.save_path, 'collector_log.txt'))
    avg_img = np.load('data/avg_img.npy')
    std_img = np.load('data/std_img.npy')
    exploration = PiecewiseSchedule([(0, 1.0), (args.epsilon_frames, 0.05), (args.num_steps / 2, 0.05)], outside_value = 0.05)

    net = ConvLSTMMulti(3, 3, args.num_total_act, True, \
                    multi_info = False, \
                    with_posinfo = args.with_posinfo, \
                    use_pos_class = args.use_pos_class, \
                    with_speed = args.with_speed, \
                    with_pos = args.with_pos, \
                    frame_history_len = args.frame_history_len)
    if torch.cuda.is_available():
        net = net.cuda()
    for param in net.parameters():
        param.requires_grad = False
    try:
        model_path = sorted(os.listdir(os.path.join(args.save_path, 'model')))[-2]
        print('Data collector: Loading model from %s' % model_path)
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)
        net_is_available = True
    except:
        net_is_available = False
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    weight = [0.97**i for i in range(args.pred_step)]
    weight = Variable(torch.from_numpy(np.array(weight).reshape((1, args.pred_step, 1))).type(dtype))
    num_off, num_coll, epi_coll, epi_off, epi_len, rewards, epi_rewards = 0, 0, 0, 0, 0, 0, []

    env = create_env(args.env, reward_ben = True, config = 'quickrace_discrete_single.xml', rescale = False)
    _ = env.reset()
    action = 1
    obs, reward, done, info = env.step(action)
    obs = resize_and_normalize(obs, args, avg_img, std_img)
    speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = args.use_pos_class) # 1x2, 1x19, 1x3
    last_obs_all = [np.zeros((avg_img.shape[2], avg_img.shape[0], avg_img.shape[1]))] * args.frame_history_len # [3x256x256]x3

    for t in range(args.buffer_size - len(os.listdir(os.path.join(args.save_path, 'dataset')))):
        last_obs_all = last_obs_all[1:] + [obs]
        this_obs_np = np.concatenate(last_obs_all, 0) # 9x256x256
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0).type(dtype)) # 1x9x256x256
        sp_var = Variable(torch.from_numpy(speed_np).type(dtype)) # 1x2
        pos_var = Variable(torch.from_numpy(pos_np).type(dtype)) # 1x19
        posxyz_var = Variable(torch.from_numpy(posxyz_np).type(dtype)) # 1x3

        if (t + 1) % args.target_update_freq == 0:
            try:
                model_path = sorted(os.listdir(os.path.join(args.save_path, 'model')))[-2]
                logger.info('Data collector: Loading model from %s' % model_path)
                state_dict = torch.load(model_path)
                net.load_state_dict(state_dict)
                net_is_available = True
            except:
                net_is_available = False

        if net_is_available and random.random() >= exploration.value(t):
            if pred_step > 1:
                action, _, _ = net.sample_action(obs_var, prev_action = action, speed=sp_var, pos=pos_var, posxyz=posxyz_var, num_time=args.pred_step, batch_step=args.batch_step, hand=False)
            else:
                action, _, _ = net.sample_action(obs_var, prev_action = action, speed=sp_var, pos=pos_var, posxyz=posxyz_var, num_time=1, batch_step=6, hand=False)
        else:
            action = np.random.randint(args.num_total_act)

        prev_info = copy.deepcopy(info)

        epi_len += 1
        obs, reward, real_done, info = env.step(int(action))
        obs = resize_and_normalize(obs, args, avg_img, std_img)
        dist = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))
        if reward <= -2.5 and abs(info['trackPos']) < 7:
            reward = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])) - np.abs(info['trackPos']) / 9.0) / 40.0
        if reward > -2.5:
            rewards += reward
        done = real_done or reward <= -2.5 or epi_len > args.max_episode_length
        logger.info('step = %d, action = %d, pos = %f, dist = %f, posX = %f, posY = %f, posZ = %f' % (epi_len, action, info['trackPos'], dist, info['pos'][0], info['pos'][1], info['pos'][2]))
        
        speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = args.use_pos_class) # 1x2, 1x19, 1x3
        offroad_flag = int(info['trackPos'] >= 5 or info['trackPos'] <= -1)
        coll_flag = int(reward == -2.5 or abs(info['trackPos']) > 7)
        epi_off += offroad_flag
        num_off += offroad_flag
        epi_coll += coll_flag
        num_coll += coll_flag

        memory_pool.store_data(obs, action, done, coll_flag, offroad_flag, prev_info['speed'], prev_info['angle'], prev_info['trackPos'])
        if done:
            last_obs_all = [np.zeros((avg_img.shape[2], avg_img.shape[0], avg_img.shape[1]))] * args.frame_history_len # [3x256x256]x3
            epi_rewards.append(rewards)
            obs = env.reset()
            action = 1
            obs, reward, done, info = env.step(action)
            obs = resize_and_normalize(obs, args.normalize, avg_img, std_img)
            speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = args.use_pos_class) 
            # print('past 100 episode rewards is ', "{0:.3f}".format(np.mean(epi_rewards[-100:])),' std is ', "{0:.15f}".format(np.std(epi_rewards[-100:])))
            logger.info('episode length: %d, reward: %f, std: %f, collision: %f, offroad: %f ' % (epi_len, np.mean(epi_rewards[-10:]), np.std(epi_rewards[-10:]), epi_coll / epi_len, epi_off / epi_len))
            # print('num coll is', num_coll, 'num off is ', num_off)
            epi_len, epi_off, epi_coll, rewards = 0, 0, 0, 0

    env.close()