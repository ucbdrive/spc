from model import *
import torch
import gym
import cv2
import pickle as pkl
from utils import *
import os
import torch.optim as optim
import torch.nn as nn
import copy
import pdb
import random
import numpy as np
from scipy.misc import imsave
from scipy.misc.pilutil import imshow
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from mpc_utils import *
from utils import make_dir, load_model, get_info_np, get_info_ls
from dqn_utils import *


def train_policy(args,
                 env,
                 num_steps=4000000, # number of training steps 
                 batch_size = 7, #batch size
                 pred_step = 10, #number of prediction step
                 normalize = True, # whether to normalize images or not
                 start_step = 100,
                 buffer_size = 50000,
                 save_path = 'model',
                 save_freq = 10, # model saving frequency
                 frame_history_len = 3,
                 num_total_act = 6):
    # prepare and start environment
    obs = env.reset()
    obs, reward, done, info = env.step(1)
    prev_info = copy.deepcopy(info)
    obs = cv2.resize(obs, (256, 256))
    train_net = ConvLSTMMulti(num_total_act, pretrain = True, frame_history_len = frame_history_len)
    train_net = train_net.cuda()

    doneCond = DoneCondition(20)
    params = [param for param in train_net.parameters() if param.requires_grad]
    optimizer = optim.Adam(params, lr = args.lr)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.05),
            (num_steps/2, 0.05),
        ], outside_value=0.05
    )

    epi_rewards, rewards = [], 0.0
    train_net, epoch, optimizer = load_model(args.save_path, train_net, data_parallel = True, optimizer=optimizer, resume=args.resume)
    train_net.train()
    
    img_buffer = IMGBuffer(1000)
    obs_buffer = ObsBuffer(frame_history_len)
    img_buffer.store_frame(obs)
    avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std(gpu=0)
    speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)

    mpc_buffer = MPCBuffer(buffer_size, frame_history_len, pred_step, num_total_act)
    prev_act, explore, epi_len = 1, 0.15, 0
 
    if args.resume:
        num_imgs_start = max(int(open(args.save_path+'/log_train_torcs.txt').readlines()[-1].split(' ')[1])-3000,0)
    else:
        num_imgs_start = 0

    if args.same_step:
        all_actions = pkl.load(open('acts_samestep.pkl','rb'))
    else:
        all_actions = pkl.load(open('acts_nosamestep.pkl','rb'))

    explore = 0.1
    last_ends = np.array([900, 200, 2])
    for tt in range(num_imgs_start, num_steps):
        ret = mpc_buffer.store_frame(obs)
        this_obs_np = obs_buffer.store_frame(obs, avg_img, std_img)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0)).float().cuda(0) 
        current_pos = np.array([info['pos'][0], info['pos'][1], info['pos'][2]])
        dist_to_pos = np.sqrt(np.sum((current_pos - last_ends)**2))
        explore = max(0.1, (800-dist_to_pos)*0.0004)
        if tt < args.epsilon_frames:
            explore = exploration.value(tt)
        rand_num = random.random()
        if rand_num <= 1-explore:
            with torch.no_grad():
                train_net.eval()
                action,_,_ = sample_action(train_net, obs_var, prev_action=prev_act, num_time=pred_step, batch_step=args.batch_step, same_step=args.same_step, all_actions=all_actions[prev_act,:,:,:])
        else:
            action = np.random.randint(num_total_act)
        obs, reward, real_done, info = env.step(int(action))
        
        reward = info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle']))-np.abs(info['trackPos'])/9.0)/40.0
        dist_this = info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle']))-np.abs(info['trackPos'])/9.0)
        prev_act = action
        done = doneCond.isdone(info['trackPos'], dist_this, info['pos']) or epi_len > 1000
        print('step ', epi_len, 'action ', action, 'pos', info['trackPos'], ' dist ', dist_this, info['pos'], 'explore', explore)
        obs = cv2.resize(obs, (256,256))
        speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)
        offroad_flag = int(info['trackPos']>=5 or info['trackPos']<=-1)
        coll_flag = int(reward==-2.5 or abs(info['trackPos'])>7)
        speed_list, pos_list = get_info_ls(prev_info)
        mpc_buffer.store_effect(ret, action, done, coll_flag, offroad_flag, speed_list[0], speed_list[1], pos_list[0])
        rewards += reward
        epi_len += 1
        if tt % 100 == 0:
            avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std()
        if done:
            last_ends = np.array([info['pos'][0], info['pos'][1], info['pos'][2]])
            obs_buffer.clear()
            epi_rewards.append(rewards)
            obs = env.reset()
            obs, reward, done, info = env.step(1)
            prev_act = 1
            obs = cv2.resize(obs, (256,256))
            speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class=False) 
            print('past 100 episode rewards is ', "{0:.3f}".format(np.mean(epi_rewards[-100:])),' std is ', "{0:.15f}".format(np.std(epi_rewards[-100:])))
            with open(args.save_path+'/log_train_torcs.txt', 'a') as fi:
                fi.write('step '+str(tt)+' reward '+str(np.mean(epi_rewards[-10:]))+' std '+str(np.std(epi_rewards[-10:]))+'\n')
            epi_len, rewards = 0, 0
        prev_info = copy.deepcopy(info) 

        # start training
        if tt % args.learning_freq == 0 and tt > args.learning_starts and mpc_buffer.can_sample(batch_size):
            print('start training') 
            sign = True
            num_epoch = 0
            while sign:
                train_net.train()
                optimizer.zero_grad()
                loss, coll_acc, off_acc, total_dist_ls = train_model(train_net, mpc_buffer, batch_size, epoch, avg_img_t, std_img_t, pred_step)
                loss.backward()
                optimizer.step()
       
                # log loss
                if epoch % 20 == 0:
                    with open(args.save_path+'/log_train.txt', 'a') as fi:
                        fi.write('collacc '+'{0:.3f}'.format(coll_acc)+' offacc '+'{0:.3f}'.format(off_acc)+\
                                ' distls '+'{0:.3f}'.format(total_dist_ls)+'\n')
                epoch+=1
                num_epoch += 1
                if num_epoch >= 10:
                    sign = False
                if epoch % save_freq == 0:
                    os.rename(args.save_path+'/model/pred_model_'+str(0).zfill(9)+'.pt', args.save_path+'/model/pred_model_'+str(0).zfill(9)+'.pt.old')
                    os.rename(args.save_path+'/optimizer/optim_'+str(0).zfill(9)+'.pt', args.save_path+'/optimizer/optim_'+str(0).zfill(9)+'.pt.old')
                    torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(0).zfill(9)+'.pt')
                    torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optim_'+str(0).zfill(9)+'.pt')
                    pkl.dump(epoch, open(args.save_path+'/epoch.pkl', 'wb'))
    return net 
