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
from torch.utils.data import Dataset, DataLoader

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
                 num_total_act = 2):
    # prepare and start environment
    obs = env.reset()
    dqn_obs = copy.deepcopy(obs)
    obs, reward, done, info = env.step(np.array([1.0, 0.0, 0.0]))
    prev_info = copy.deepcopy(info)
    obs = cv2.resize(obs, (256, 256))
    train_net = ConvLSTMMulti(num_total_act, pretrain = True, frame_history_len = frame_history_len)
    train_net = train_net.cuda()
    net = ConvLSTMMulti(num_total_act, pretrain=True, frame_history_len = frame_history_len)

    # dqn net
    dqn_net = atari_model(3 * frame_history_len, 11, frame_history_len).cuda().float()
    dqn_net.train()
    target_q_net = atari_model(3 * frame_history_len, 11, frame_history_len).cuda().float()
    target_q_net.eval()
    dqn_optimizer = optim.Adam(dqn_net.parameters(), lr=args.lr, amsgrad=True)
    replay_buffer = ReplayBuffer(100000, frame_history_len)
    if args.resume:
        target_q_net.load_state_dict(torch.load(args.save_path+'/model/dqn.pt'))
        dqn_net.load_state_dict(torch.load(args.save_path+'/model/dqn.pt'))
    
    def select_epilson_greedy_action(model, obs, t, exploration, num_actions):
        with torch.no_grad():
            sample = random.random()
            eps_threshold = exploration.value(t)
            if sample > eps_threshold:
                obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
                action = int(model(Variable(obs, requires_grad=False)).data.max(1)[1].cpu().numpy())
            else:
                action = random.randrange(num_actions)
        return int(action)

    doneCond = DoneCondition(30)
    params = [param for param in train_net.parameters() if param.requires_grad]
    optimizer = optim.Adam(params, lr = args.lr, amsgrad=True)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
    )

    dqn_explore = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
    )

    epi_rewards, rewards = [], 0.0
    train_net, epoch, optimizer = load_model(args.save_path, train_net, data_parallel = True, optimizer=optimizer, resume=args.resume)
    train_net.train()
    net.load_state_dict(train_net.module.state_dict())
    net.float().cuda()
  
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
 
    img_buffer = IMGBuffer(1000)
    obs_buffer = ObsBuffer(frame_history_len)
    img_buffer.store_frame(obs)
    avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std(gpu=0)
    speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)

    mpc_buffer = MPCBuffer(buffer_size, frame_history_len, pred_step, num_total_act, continuous=True)
    prev_act, explore, epi_len = np.array([1.0, 0.0]), 0.15, 0
 
    if args.resume:
        num_imgs_start = max(int(open(args.save_path+'/log_train_torcs.txt').readlines()[-1].split(' ')[1])-1000,0)
    else:
        num_imgs_start = 0

    num_param_updates = 0
    explore = 0.1
    rewards2 = 0
    epi_rewards2 = []
    for tt in range(num_imgs_start, num_steps):
        dqn_ret = replay_buffer.store_frame(cv2.resize(dqn_obs, (84, 84)))
        dqn_net_obs = replay_buffer.encode_recent_observation()
        dqn_action = select_epilson_greedy_action(dqn_net, dqn_net_obs, tt, dqn_explore, 11)
         
        ret = mpc_buffer.store_frame(obs)
        this_obs_np = obs_buffer.store_frame(obs, avg_img, std_img)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0)).float().cuda(0) 
        explore = exploration.value(tt)
        rand_num = random.random()
        if rand_num <= 1-explore:
            action = sample_cont_action(net, obs_var, prev_action=prev_act, num_time=pred_step) 
        else:
            action = (np.random.rand(2)*2-1)
        action = np.clip(action, -1.0, 1.0)
        if abs(action[1]) <= (dqn_action) * 0.1:
            action[1] = 0
        exe_action = np.zeros(3)
        exe_action[0] = np.abs(action[0])
        exe_action[2] = action[1]*0.1
        obs, reward, real_done, info = env.step(exe_action)

 
        dist_this = info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle']))-np.abs(info['trackPos'])/9.0)
        reward = info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle']))-np.abs(info['trackPos'])/9.0)/40.0
        reward2 = info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle'])))/40.0
        done = doneCond.isdone(info['trackPos'], dist_this, info['pos']) or epi_len > 1000
        prev_act = action
        print('step ', epi_len, 'action ', "{0:.3f}".format(exe_action[0]), "{0:.3f}".format(exe_action[2]), 'pos', "{0:.3f}".format(info['trackPos']), ' dist ', "{0:.3f}".format(dist_this), np.round(info['pos']), "{0:.3f}".format(info['angle']), 'explore', "{0:.3f}".format(explore))
        obs = cv2.resize(obs, (256,256))
        speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)
        offroad_flag = int(info['trackPos']>=3 or info['trackPos']<=-1)
        coll_flag = int(reward==-2.5 or abs(info['trackPos'])>7)
        speed_list, pos_list = get_info_ls(prev_info)
        mpc_buffer.store_effect(ret, action, done, coll_flag, offroad_flag, speed_list[0], speed_list[1], pos_list[0])
        rewards += reward
        rewards2 += reward2
        epi_len += 1
        if tt % 100 == 0:
            avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std()

        if done:
            obs_buffer.clear()
            epi_rewards.append(rewards)
            obs = env.reset()
            epi_rewards2.append(rewards2)
            obs, reward, done, info = env.step(np.array([1.0, 0.0, 0.0]))
            prev_act = np.array([1.0, 0.0])
            obs = cv2.resize(obs, (256,256))
            speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class=False) 
            print('past 100 episode rewards is ', "{0:.3f}".format(np.mean(epi_rewards[-100:])),' std is ', "{0:.15f}".format(np.std(epi_rewards[-100:])))
            with open(args.save_path+'/log_train_torcs.txt', 'a') as fi:
                fi.write('step '+str(tt)+' reward '+str(np.mean(epi_rewards[-10:]))+' std '+str(np.std(epi_rewards[-10:]))+\
                        ' reward2 '+str(np.mean(epi_rewards2[-10:]))+' std '+str(np.std(epi_rewards2[-10:]))+'\n')
            epi_len, rewards, rewards2 = 0, 0, 0
        prev_info = copy.deepcopy(info) 
        replay_buffer.store_effect(dqn_ret, dqn_action, reward, done)
        dqn_obs = copy.deepcopy(obs)

        # start training
        if tt % args.learning_freq == 0 and tt > args.learning_starts and mpc_buffer.can_sample(batch_size):
            print('start training') 
            sign = True
            num_epoch = 0
            while sign:
                optimizer.zero_grad()
                loss, coll_acc, off_acc, total_dist_ls = train_model(train_net, mpc_buffer, batch_size, epoch, avg_img_t, std_img_t, pred_step)
                loss.backward()
                optimizer.step()

                #optimizer.zero_grad()
                #loss = train_model_imitation(train_net, mpc_buffer, batch_size, epoch, avg_img_t, std_img_t, pred_step)
                #loss.backward()
                #optimizer.step()

                net.load_state_dict(train_net.module.state_dict()) 

                # log loss
                if epoch % 200 == 0:
                    if total_dist_ls > 1000:
                        for g in optimizer.param_groups:
                            g['lr'] = 0.001
                    with open(args.save_path+'/log_train.txt', 'a') as fi:
                        fi.write('collacc '+'{0:.3f}'.format(coll_acc)+' offacc '+'{0:.3f}'.format(off_acc)+\
                                ' distls '+'{0:.3f}'.format(total_dist_ls)+'\n')
                epoch+=1
                num_epoch += 1
                if num_epoch >= 10 and total_dist_ls < 100:
                    sign = False
                if epoch % save_freq == 0:
                    try:
                        os.rename(args.save_path+'/model/pred_model_'+str(0).zfill(9)+'.pt', args.save_path+'/model/pred_model_'+str(0).zfill(9)+'.pt.old')
                        os.rename(args.save_path+'/optimizer/optim_'+str(0).zfill(9)+'.pt', args.save_path+'/optimizer/optim_'+str(0).zfill(9)+'.pt.old')
                    except:
                        pass
                    torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(0).zfill(9)+'.pt')
                    torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optim_'+str(0).zfill(9)+'.pt')
                    pkl.dump(epoch, open(args.save_path+'/epoch.pkl', 'wb'))
            
            obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)
            obs_t_batch     = Variable(torch.from_numpy(obs_t_batch).type(dtype) / 255.0)
            act_batch       = Variable(torch.from_numpy(act_batch).long()).cuda()
            rew_batch       = Variable(torch.from_numpy(rew_batch)).cuda()
            obs_tp1_batch   = Variable(torch.from_numpy(obs_tp1_batch).type(dtype) / 255.0)
            done_mask       = Variable(torch.from_numpy(done_mask)).type(dtype)
           
            q_a_values = dqn_net(obs_t_batch).gather(1, act_batch.unsqueeze(1))
            q_a_values_tp1 = target_q_net(obs_tp1_batch).detach().max(1)[0]
            target_values = rew_batch + (0.99 * (1-done_mask) * q_a_values_tp1)
            dqn_loss = ((target_values.view(q_a_values.size()) - q_a_values)**2).mean()
            dqn_optimizer.zero_grad()
            dqn_loss.backward()
            dqn_optimizer.step()
            num_param_updates += 1
        
            if num_param_updates % 100==0:
                target_q_net.load_state_dict(dqn_net.state_dict())
                torch.save(target_q_net.state_dict(), args.save_path+'/model/dqn.pt') 
    return net 
