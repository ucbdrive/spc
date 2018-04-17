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
from collections import OrderedDict

def make_dir(path):
    print('make ', path)
    os.mkdir(path)
    return

def load_model(path):
    file_list = os.listdir(path)
    file_list = sorted(file_list)
    return file_list[-2]


def get_info_np(info, use_pos_class=False):
    speed_np = np.array([[info['speed'], info['angle']]]).reshape((1,2))
    if use_pos_class:
        pos = int(round(np.minimum(np.maximum(info['trackPos'],-9),9))+9)
        pos_np = np.zeros((1,19))
        pos_np[0, pos] = 1
    else:
        pos_np = np.array([[info['trackPos']]]).reshape((1,1))
    posxyz_np = np.array([info['pos'][0], info['pos'][1], info['pos'][2]]).reshape((1,3))
    return speed_np, pos_np, posxyz_np

def test_policy(env, num_steps, use_pos_class=False, frame_history_len=3):
    obs = env.reset()
    obs, reward, done, info = env.step(4)
    obs = cv2.resize(obs, (256,256))
    net = ConvLSTMMulti(3,3,6, False, multi_info=False, with_posinfo=False, use_pos_class=use_pos_class, with_speed=False, frame_history_len=3)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        net = net.cuda()
    else:
        dtype = torch.FloatTensor

    net.eval()
    epi_rewards = []
    rewards = 0.0
    try:
        model_path = load_model('model')
        state_dict = torch.load('model/'+model_path)
        net.load_state_dict(state_dict)
        print('load model', model_path)
        if use_cuda == False:
            net = net.cpu()
    except:
        pass
    for param in net.parameters():
        param.requires_grad = False

    avg_img = pkl.load(open('avg_img.pkl','rb'))
    std_img = pkl.load(open('std_img.pkl','rb'))
    num_imgs = 0
    prev_act = 0
    speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class=use_pos_class)
    posxyz_var = Variable(torch.from_numpy(posxyz_np).view(1,3).type(dtype))
    epi_len = 0
    last_obs_all = []
    for tt in range(num_steps):
        obs_np = (obs-avg_img)/(std_img+0.0001)
        if len(last_obs_all) < frame_history_len:
            last_obs_all = []
            for ii in range(frame_history_len):
                last_obs_all.append(obs_np.transpose(2,0,1))
        else:
            last_obs_all = last_obs_all[1:]+[obs_np.transpose(2,0,1)]
        obs_np_new = np.concatenate(last_obs_all, 0)
        obs_var = Variable(torch.from_numpy(obs_np_new).unsqueeze(0).type(dtype))
        sp_var = Variable(torch.from_numpy(speed_np).view(1,2).type(dtype))
        if use_pos_class == False:
            pos_var = Variable(torch.from_numpy(pos_np).view(1,1).type(dtype))
        else:
            pos_var = Variable(torch.from_numpy(pos_np).view(1,19).type(dtype))
        action, collprob, offprob = net.sample_action(obs_var, prev_action=prev_act, speed=sp_var, pos=pos_var, num_time=9, posxyz=posxyz_var, batch_step=100, hand=True)
        off_probs = 0
        all_off_probs = []
        all_dist = []
        all_colls = []
        for i in range(6):
            outs = net.sample_action(obs_var,prev_action=i,speed=sp_var,pos=pos_var,num_time=15,posxyz=posxyz_var,calculate_loss=True)
            all_off_probs.append(outs[1][:].reshape((-1)).sum())
            all_dist.append(outs[3].reshape((-1)).sum())
            all_colls.append(outs[0][:].reshape((-1)).sum())
            if i == 0 or i == 2 or i == 3 or i == 5:# or i == 6 or i == 8:
                if np.mean(outs[1]) > off_probs:
                    off_probs = np.mean(outs[1])
                    this_action = i
            #print('action ',i,'coll',outs[0].reshape((-1)).sum(),outs[5].reshape((-1)).sum(),'off',outs[1].reshape((-1)).sum(),outs[6].reshape((-1)).sum(),'dist',outs[3].reshape((-1)),outs[8].reshape((-1)),'posgt', info['trackPos'])
        #if abs(info['trackPos']) >=2.0:
        #    action = this_action
        # print('coll', 5-np.array(all_colls).argsort().argsort())
        # print('off', 5-np.array(all_off_probs).argsort().argsort())
        # print('dist',5- np.array(all_dist).argsort().argsort())
        off_rank = 5-np.array(all_off_probs).argsort().argsort()
        coll_rank = 5-np.array(all_colls).argsort().argsort()
        dist_rank = 5-np.array(all_dist).argsort().argsort()
        all_rank = (5-np.array(all_colls).argsort().argsort()+5-np.array(all_off_probs).argsort().argsort()+5-np.array(all_dist).argsort().argsort()).argsort().argsort()
        all_rank2 = (5-np.array(all_off_probs).argsort().argsort()+5-np.array(all_dist).argsort().argsort()).argsort().argsort()
        # print('all rank', all_rank)
        # print('all rank2 ', all_rank2)
        # if np.mean(all_off_probs) >= 0.7:
        #    action = np.array(all_dist).argsort()[0]
        # else:
        #    action = (np.array(all_colls)+np.array(all_off_probs)).argsort()[0]
        # action = 1
        sign_act = True
        # if np.mean(all_off_probs)/8.0 < 0.4:# and all_dist[1] > 14:
        #    action = np.argmin(off_rank)
        # else:
        #    action = np.argmin(all_rank2)
        # cnt = 0
        # real_action = action.copy()
        # while sign_act and cnt <= 7:
        #    if off_rank[real_action] <=4 and coll_rank[real_action]<=4 and dist_rank[real_action] <=4:
        #        sign_act = False
        #    else:
        #        if np.mean(all_off_probs)/8.0<0.4:
        #            real_action = int(np.where(off_rank.reshape((-1))==(cnt+1))[0])
        #        else:
        #            real_action = int(np.where(all_rank2.reshape((-1))==(cnt+1))[0])
        #    cnt+= 1
        # if cnt <=7:
        #    action = real_action 
        obs, reward, done, info = env.step(action)
        dist_this = float(info['speed'])*(np.cos(float(info['angle']))-np.abs(np.sin(float(info['angle'])))-0.1*np.abs(info['trackPos']))
        #print(tt, action, 'collprob', collprob[0], 'reward ', reward, 'offprob', offprob[0][0], 'posgt', info['trackPos'], 'dist ', dist_this)
        print('action', action, 'dist', dist_this, 'pos', info['trackPos'])
        prev_act = action
        done = done #info['should_reset'] or epi_len >= 2000
        epi_len += 1
        obs = cv2.resize(obs, (256,256))
        speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class=use_pos_class)
        posxyz_var = Variable(torch.from_numpy(posxyz_np).type(dtype))
        dist_this = float(info['speed'])*(np.cos(float(info['angle']))) #-np.abs(np.sin(float(info['angle']))))    
        dist_str = "{:2.4f}".format(dist_this)
        speed_str = "{:2.4f}".format(speed_np[0,0])
        angle_str = "{:3.1f}".format(speed_np[0,1]/np.pi*180)
        cv2.imwrite('test/'+str(int(tt)).zfill(9)+'_'+str(action)+'_'+str(done)+'_'+dist_str+'_'+speed_str+'_'+angle_str+'.png', obs.reshape((256,256,3)))
        rewards += reward
        if done:
            obs = env.reset()
            obs, reward, done, info = env.step(4)
            speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class=use_pos_class)
            obs = cv2.resize(obs, (256,256))
            epi_rewards.append(rewards)
            rewards = 0.0
            epi_len = 0
            state_dict = torch.load('model/'+load_model('model'))
            try:
                net.load_state_dict(state_dict)
            except:
                pass
            print('past 100 episode rewards is ', np.mean(epi_rewards[-100:]),' std is ', np.std(epi_rewards[-100:]))
            with open('log_test_torcs.txt', 'wb') as fi:
                fi.write('reward '+str(np.mean(epi_rewards[-100:]))+' std '+str(np.std(epi_rewards[-100:]))+'\n')
    return net              
