import sys
import os
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from dqn_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import PIL.Image as Image
import pdb
import cv2
import copy
import argparse
import gym
import pickle as pkl

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

def test(args,
        env,
        q_func,
        replay_buffer_size=100,
        frame_history_len=4,
        load_path=None,
        adv_path=None,
        adv_model=None):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    if args.use_cuda == True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    if len(env.observation_space.shape) == 1:
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n
    
    q_value = q_func(input_arg, num_actions).type(dtype)
    print('load model ', load_path)
    global_step = int(load_path.split('/')[-1])
    q_value.load_state_dict(torch.load(load_path+'/model_0.pt'))
    q_value.eval()
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    if args.test_with_adv_agent == True:
        adv_q_value = adv_model(input_arg, num_actions, 10).type(dtype)
        print('load adv model', adv_path)
        adv_q_value.load_state_dict(torch.load(adv_path+'/advmodel_0.pt'))
    
    def select_action(model, obs):
        obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
        action = int(model(Variable(obs, volatile=True)).data.max(1)[1].cpu().numpy())
        return torch.IntTensor([[action]])

    def select_adv_action(model, obs):
        obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255.0
        values = model(Variable(obs, volatile=True)).data.cpu().numpy()
        values = values.reshape(-1)
        values = values.argsort().argsort()
        values = values.reshape((10, 9))
        out = np.sum(values, axis=0).reshape(9,)
        action = np.argmax(out)
        return torch.IntTensor([[action]])

    def get_state_value(model, obs, num_actions, this_action=None):
        obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255.0
        values = model(Variable(obs, volatile=True)).data.cpu().numpy()
        values = values.reshape((num_actions,))
        state_value = np.max(values)
        if this_action is not None:
            state_value = values[int(this_action)]
        return state_value

    last_obs = env.reset()
    episode_rewards = []
    episode_reward = 0.0
    episode_cata_rewards = []
    episode_cata_reward = 0.0
    num_epi = 0
    num_catas = []
    num_cata = 0.0
    if args.save_img == True:       
        if args.test_without_adv == True:
            img_path = args.save_path + '/data'
            act_path = args.save_path + '/action'
            done_path = args.save_path + '/done'
            reward_path = args.save_path + '/reward'
            coll_path = args.save_path + '/coll'
            speed_path = args.save_path + '/speed'
            offroad_path = args.save_path + '/offroad'
        else:
            img_path = args.save_path + '/data_with'
            act_path = args.save_path + '/action_with'
            done_path = args.save_path + '/done_with'
            reward_path = args.save_path + '/reward_with'
            coll_path = args.save_path + '/coll_with'
            speed_path = args.save_path + '/speed_with'
            offroad_path = args.save_path + '/offroad_with'
        try: 
            os.mkdir(img_path)
        except: 
            pass
        try:
            os.mkdir(act_path)
        except: 
            pass
        try: 
            os.mkdir(done_path)
        except: 
            pass
        try:
            os.mkdir(reward_path)
        except:
            pass
        try:
            os.mkdir(coll_path)
        except:
            pass
        try:
            os.mkdir(speed_path)
        except:
            pass
        try:
            os.mkdir(offroad_path)
        except:
            pass
        num_imgs = len(os.listdir(img_path))

    action_by = 0
    act_cnt = -1
    epi_len = 0
    epi_lens = []
    for t in itertools.count():
        if args.save_img == True:
            cv2.imwrite(img_path+'/'+str(num_imgs).zfill(9)+'.png', last_obs)
            last_obs = cv2.resize(last_obs, (84, 84))
            last_obs = last_obs.reshape((84, 84, img_c))
            num_imgs += 1
        ret = replay_buffer.store_frame(last_obs)
        obs = replay_buffer.encode_recent_observation()
        action = int(select_action(q_value, obs)[0, 0])
        action_by = 0
        if act_cnt >= 10+args.test_adv_freq:
            act_cnt = 0
        if args.test_with_random_adv == True and act_cnt < args.test_adv_freq:
            action = random.randrange(num_actions)
            action_by = 1

        if args.test_with_adv_agent == True and act_cnt < args.test_adv_freq:
            action = int(select_adv_action(adv_q_value, obs)[0,0])
            action_by = 1

        act_cnt += 1
        last_obs, reward, done, info = env.step(action)
        state_value = get_state_value(q_value, obs, 9)
        state_qvalue = get_state_value(q_value, obs, 9, action)
        if reward <= -2.5:
            episode_cata_reward += reward
        else:
            episode_reward += reward
        epi_len += 1
        if reward <= -2.5:
            num_cata += 1

        if args.save_img == True:
            pkl.dump([action, action_by], open(act_path+'/'+str(num_imgs-1).zfill(9)+'.pkl', 'wb'))
            pkl.dump(done, open(done_path+'/'+str(num_imgs-1).zfill(9)+'.pkl', 'wb'))
            pkl.dump(reward, open(reward_path+'/'+str(num_imgs-1).zfill(9)+'.pkl','wb'))
            if abs(info['trackPos']) <= info['trackWidth']/2.0:
                offroad = 0
            else:
                offroad = 1
            pkl.dump(offroad, open(offroad_path+'/'+str(num_imgs-1).zfill(9)+'.pkl','wb'))
            pkl.dump([info['speed'], info['angle']], open(speed_path+'/'+str(num_imgs-1).zfill(9)+'.pkl','wb')) 
            if reward <= -2.4:
                pkl.dump(1, open(coll_path+'/'+str(num_imgs-1).zfill(9)+'.pkl','wb'))
            else:
                pkl.dump(0, open(coll_path+'/'+str(num_imgs-1).zfill(9)+'.pkl','wb'))
        
        this_done0 = info['should_reset']
        with open(os.path.join(args.save_path, args.log_name+'.rollout'), 'a') as fi:
            fi.write('step '+str(global_step)+' '+str(t)+' action '+str(action)+\
                    ' value '+str(state_value)+' qvalue '+str(state_qvalue)+\
                    ' reward '+str(reward)+' done '+str(int(this_done0))+'\n')

        if info['should_reset']:
            last_obs = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            episode_cata_rewards.append(episode_cata_reward)
            episode_cata_reward = 0.0
            num_epi += 1
            num_catas.append(num_cata/(epi_len*1.0))
            num_cata = 0
            epi_lens.append(epi_len)
            epi_len = 0
            print('finish one episode')
            if num_epi >= 10:
                break
        replay_buffer.store_effect(ret, action, reward, done)
    
    print('step ', global_step, ' reward ', np.mean(episode_rewards), \
            ' std ', np.std(episode_rewards), ' catareward ', np.mean(episode_cata_rewards), ' std ', np.std(episode_cata_rewards), \
            ' num cata ', np.mean(num_catas), np.std(num_catas), \
            ' epi len ', np.sum(epi_lens))
    with open(os.path.join(args.save_path, args.log_name), 'a') as fi:
        fi.write('step '+str(global_step)+\
                ' mean reward '+str(np.mean(episode_rewards))+\
                ' var reward '+str(np.std(episode_rewards))+\
                ' mean catareward '+str(np.mean(episode_cata_rewards))+\
                ' var catareward '+str(np.std(episode_cata_rewards))+\
                ' cata '+\
                str(np.mean(num_catas))+' '+str(np.std(num_catas))+\
                str('epi len is ')+str(np.sum(epi_lens))+'\n')
    sys.stdout.flush()
