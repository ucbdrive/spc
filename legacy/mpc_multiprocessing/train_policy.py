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
                 use_pos_class = False,
                 save_path = 'model',
                 save_freq = 10, # model saving frequency
                 with_speed = False,
                 with_posinfo = False,
                 with_pos = False,
                 frame_history_len = 3,
                 num_total_act = 6):
    # prepare and start environment
    obs = env.reset()
    obs, reward, done, info = env.step(1)
    prev_info = copy.deepcopy(info)
    obs = cv2.resize(obs, (256, 256))
    use_cuda = torch.cuda.is_available()
    net = ConvLSTMMulti(3, 3, num_total_act, True, \
                multi_info = False, \
                with_posinfo = with_posinfo, \
                use_pos_class = use_pos_class, \
                with_speed = with_speed, \
                with_pos = with_pos, \
                frame_history_len = frame_history_len,
                with_dla=args.with_dla)
    train_net = ConvLSTMMulti(3, 3, num_total_act, True, \
                multi_info = False, \
                with_posinfo = with_posinfo, \
                use_pos_class = use_pos_class, \
                with_speed = with_speed, \
                with_pos = with_pos, \
                frame_history_len = frame_history_len,
                with_dla=args.with_dla)
    if use_cuda:
        net = net.cuda(0)
        net.conv_lstm.dla.cuda(0)
        train_net = train_net.cuda()

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
    net, epoch = load_model(args.save_path+'/model', net, data_parallel = False)
    net.eval()
    net.cuda(0)
    for param in net.parameters():
        param.requires_grad = False

    train_net, epoch = load_model(args.save_path+'/model', train_net, data_parallel = True)
    train_net.train()

    img_buffer = IMGBuffer(1000)
    img_buffer.store_frame(obs)
    avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std(gpu=0)
    speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = use_pos_class)

    last_obs_all = []
    mpc_buffer = MPCBuffer(buffer_size, frame_history_len, pred_step, num_total_act)
    prev_act, num_off, num_coll, epi_coll, epi_off, explore = 1, 0, 0, 0, 0, 0.15
    weight = [0.97**i for i in range(pred_step)]
    weight = Variable(torch.from_numpy(np.array(weight).reshape((1, pred_step, 1))).type(dtype))
    epi_len, off_cnt = 0, 0 
    
    try:
        num_imgs_start = max(int(open(args.save_path+'/log_train_torcs.txt').readlines()[-1].split(' ')[1])-300,0)
    except:
        num_imgs_start = 0
    for tt in range(num_imgs_start, num_steps):
        ret = mpc_buffer.store_frame(obs)
        if normalize:
            obs_np = (obs-avg_img)/(std_img+0.0001)
        else:
            obs_np = obs/255.0
        obs_np = obs_np.transpose(2,0,1)
        if len(last_obs_all) < frame_history_len:
            last_obs_all = []
            for ii in range(frame_history_len):
                last_obs_all.append(obs_np)
        else:
            last_obs_all = last_obs_all[1:]+[obs_np]

        this_obs_np = np.concatenate(last_obs_all, 0)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0)).float().cuda(0)
        sp_var = Variable(torch.from_numpy(speed_np).view(1,2)).float().cuda(0)
        pos_var = Variable(torch.from_numpy(pos_np)).float().cuda(0)
        posxyz_var = Variable(torch.from_numpy(posxyz_np).view(1,3)).float().cuda(0)
        explore = exploration.value(tt)
        rand_num = random.random()
        if rand_num <= 1-explore and ((epi_len % 1 == 0 and pred_step == 15) or (epi_len % 3 == 0 and pred_step == 12)):
            if pred_step > 1:
                action,_,_ = net.sample_action(obs_var, prev_action=prev_act, speed=sp_var, pos=pos_var, posxyz=posxyz_var, num_time=pred_step, batch_step=args.batch_step, hand=False, gpu=0)
            else:
                action,_,_ = net.sample_action(obs_var, prev_action=prev_act, speed=sp_var, pos=pos_var, posxyz=posxyz_var, num_time=1, batch_step=6, hand=False)
        elif (epi_len % 1 == 0 and pred_step == 15) or (epi_len % 3 == 0 and pred_step == 12):
            action = np.random.randint(num_total_act)
        else:
            action = prev_act 
        obs, reward, real_done, info = env.step(int(action))
        #img_buffer.store_frame(cv2.resize(obs, (256,256)))
        #if tt % 100 == 0:
        #    avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std(gpu=0)
        reward = info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle']))-np.abs(info['trackPos'])/9.0)/40.0
        dist_this = info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle']))-np.abs(info['trackPos'])/7.0)
        prev_act = action
        if info['trackPos'] <= -6.2 and info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle']))) < 1:
            off_cnt += 1
        if info['trackPos'] > -6.2:
            off_cnt = 0
        if reward <= -2.5 and abs(info['trackPos']) < 7:
            reward = info['speed']*(np.cos(info['angle'])-np.abs(np.sin(info['angle']))-np.abs(info['trackPos'])/9.0)/40.0
            done = False
        if info['trackPos'] <= -9 or info['trackPos'] >= 15.0 or off_cnt >= 20:
            done = True
        done = done or epi_len > 1000
        
        print('step ', epi_len, 'action ', action, 'pos', info['trackPos'], ' dist ', dist_this, info['pos'])
        obs = cv2.resize(obs, (256,256))
        speed_np, pos_np, posxyz_np = get_info_np(info,use_pos_class=use_pos_class)
        offroad_flag = int(info['trackPos']>=5 or info['trackPos']<=-1)
        coll_flag = int(reward==-2.5 or abs(info['trackPos'])>7)
        if offroad_flag:
            epi_off += 1
        if coll_flag:
            epi_coll += 1
        speed_list, pos_list = get_info_ls(prev_info)
        mpc_buffer.store_effect(ret, action, done, coll_flag, offroad_flag, speed_list[0], speed_list[1], pos_list[0])
        num_coll += coll_flag
        num_off += offroad_flag
        if reward >-2.5:
            rewards += reward
        epi_len += 1
        if tt % args.target_update_freq == 0:
            net, _ = load_model(args.save_path+'/model', net, data_parallel=False)
            net.eval()
            net.cuda(0)
            for param in net.parameters():
                param.requires_grad = False
        if done:
            last_obs_all = []
            epi_rewards.append(rewards)
            obs = env.reset()
            obs, reward, done, info = env.step(1)
            prev_act = 1
            obs = cv2.resize(obs, (256,256))
            speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class=use_pos_class) 
            print('past 100 episode rewards is ', "{0:.3f}".format(np.mean(epi_rewards[-100:])),' std is ', "{0:.15f}".format(np.std(epi_rewards[-100:])))
            with open(args.save_path+'/log_train_torcs.txt', 'a') as fi:
                fi.write('step '+str(tt)+' reward '+str(np.mean(epi_rewards[-10:]))+' std '+str(np.std(epi_rewards[-10:]))+\
                        ' epicoll '+str(epi_coll/(epi_len*1.0))+ ' epioff '+str(epi_off/(epi_len*1.0))+'\n')
            print('num coll is', num_coll, 'num off is ', num_off)
            epi_len, epi_off, epi_coll, rewards = 0, 0, 0, 0
        prev_info = copy.deepcopy(info) 

        # start training
        if tt % args.learning_freq == 0 and tt > args.learning_starts and num_coll >= 1 and num_off >= 10 and mpc_buffer.can_sample(batch_size):
            print('start training') 
            sign = True
            num_epoch = 0
            while sign:
                num_batch, all_label, all_pred, off_label, off_pred, prev_loss, speed_loss = 0, [], [], [], [], 10000.0, 0
                total_loss = 0
                total_coll_ls = 0
                total_off_ls = 0
                total_pred_ls = 0
                total_pos_ls = 0
                total_dist_ls = 0
                total_posxyz_ls = 0
                
                if epoch % 20 == 0:
                    x, idxes = mpc_buffer.sample(batch_size, sample_early = True)
                else:
                    x, idxes = mpc_buffer.sample(batch_size, sample_early = False)
                x = list(x)
                for iii in range(len(x)):
                    x[iii] = torch.from_numpy(x[iii]).type(dtype)
                act_batch = Variable(x[0], requires_grad=False).type(dtype)
                coll_batch = Variable(x[1], requires_grad=False).type(dtype)
                speed_batch = Variable(x[2], requires_grad=False).type(dtype)
                offroad_batch = Variable(x[3], requires_grad=False).type(dtype)
                dist_batch = Variable(x[4]).type(dtype)
                if normalize:
                    img_batch = Variable(((x[5].float()-avg_img_t)/(std_img_t+0.0001)).type(dtype), requires_grad=False)
                    nximg_batch = Variable(((x[6].float()-avg_img_t)/(std_img_t+0.0001)).type(dtype), requires_grad=False)
                else:
                    img_batch = Variable(x[5], requires_grad=False).type(dtype)/255.0
                    nximg_batch = Variable(x[6], requires_grad=False).type(dtype)/255.0
                pos_batch = Variable(x[7]).type(dtype)
                posxyz_batch = None
                optimizer.zero_grad()
                pred_coll, pred_enc, pred_off, pred_sp, pred_dist,pred_pos,pred_posxyz,_,_ = train_net(img_batch, act_batch, speed_batch, pos_batch, int(act_batch.size()[1]), posxyz=posxyz_batch)
                with torch.no_grad():
                    nximg_enc = train_net(nximg_batch, get_feature=True)
                    nximg_enc = nximg_enc.detach()
                # calculating loss function
                coll_ls = Focal_Loss(pred_coll.view(-1,2), (torch.max(coll_batch.view(-1,2),-1)[1]).view(-1), reduce=False)
                offroad_ls = Focal_Loss(pred_off.view(-1,2), (torch.max(offroad_batch.view(-1,2),-1)[1]).view(-1), reduce=False)
                pos_batch2 = pos_batch[:,1:,:]
                pos_batch2 = pos_batch2.contiguous()
                pos_batch2 = pos_batch2.view(-1,1)
                if with_speed:
                    speed_ls = nn.L1Loss()(pred_sp, speed_batch[:,1:,:])
                    speed_loss += float(speed_ls.data.cpu().numpy())
                else:
                    speed_ls = 0
                dist_ls = nn.MSELoss(reduce=False)(pred_dist.view(-1,pred_step), dist_batch[:,1:].view(-1,pred_step))
                dist_ls = dist_ls.view(-1)
                if use_pos_class == False and with_pos:
                    this_weight = weight.repeat(int(pos_batch.size()[0]), 1,1)
                    pred_pos = pred_pos * this_weight
                    pos_batch = pos_batch[:,1:,:] * this_weight
                    pos_ls = nn.L1Loss()(pred_pos, pos_batch)
                elif with_pos:
                    pos_batch = pos_batch[:,1:,:].contiguous()
                    pos_ls = Focal_Loss(pred_pos.view(-1,19), (torch.max(pos_batch.view(-1,19),-1)[1]).view(-1))
                pred_ls = nn.L1Loss(reduce=False)(pred_enc, nximg_enc).sum(-1).view(-1)
                loss = pred_ls + coll_ls + offroad_ls + dist_ls# + 20*pos_ls 
                loss_np = loss.data.cpu().numpy().reshape((-1, pred_step)).sum(-1)
                mpc_buffer.store_loss(loss_np, idxes)
        
                ''' loss '''
                coll_ls = Focal_Loss(pred_coll.view(-1,2), (torch.max(coll_batch.view(-1,2),-1)[1]).view(-1), reduce=True)
                offroad_ls = Focal_Loss(pred_off.view(-1,2), (torch.max(offroad_batch.view(-1,2),-1)[1]).view(-1), reduce=True)
                dist_ls = nn.MSELoss()(pred_dist.view(-1,pred_step), dist_batch[:,1:].view(-1,pred_step))
                pred_ls = nn.L1Loss()(pred_enc, nximg_enc).sum()
                loss = pred_ls + coll_ls + offroad_ls + dist_ls
                loss.backward()
                optimizer.step()
        
                # log loss
                total_coll_ls += float(coll_ls.data.cpu().numpy())
                total_off_ls += float(offroad_ls.data.cpu().numpy())
                total_pred_ls += float(pred_ls.data.cpu().numpy())
                total_dist_ls += float(dist_ls.data.cpu().numpy()) 
                total_loss += float(loss.data.cpu().numpy())

                pred_coll_np = pred_coll.view(-1,2).data.cpu().numpy()
                coll_np = coll_batch.view(-1,2).data.cpu().numpy()
                pred_coll_np = np.argmax(pred_coll_np, 1)
                coll_np = np.argmax(coll_np, 1)
                all_label.append(coll_np)
                all_pred.append(pred_coll_np)
                pred_off_np = pred_off.view(-1,2).data.cpu().numpy()
                off_np = offroad_batch.view(-1,2).data.cpu().numpy()
                pred_off_np = np.argmax(pred_off_np, 1)
                off_np = np.argmax(off_np, 1)
                off_label.append(off_np)
                off_pred.append(pred_off_np)
                if float(loss.data.cpu().numpy()) < prev_loss:
                    prev_loss = float(loss.data.cpu().numpy())
                all_label = np.concatenate(all_label)
                all_pred = np.concatenate(all_pred)
                off_label = np.concatenate(off_label)
                off_pred = np.concatenate(off_pred)
                cnf_matrix = confusion_matrix(all_label, all_pred)
                cnf_matrix_off = confusion_matrix(off_label, off_pred)
                num_batch += 1
                try:
                    coll_acc = (cnf_matrix[0,0]+cnf_matrix[1,1])/(cnf_matrix.sum()*1.0)
                    off_accuracy = (cnf_matrix_off[0,0]+cnf_matrix_off[1,1])/(cnf_matrix_off.sum()*1.0)
                    if epoch % 20 == 0:
                        print('sample early collacc', "{0:.3f}".format(coll_acc), \
                            "{0:.3f}".format(total_coll_ls/num_batch), \
                            'offacc', "{0:.3f}".format(off_accuracy), \
                            "{0:.3f}".format(total_off_ls/num_batch), \
                            'spls', "{0:.3f}".format(speed_loss/num_batch), 'ttls', "{0:.3f}".format(total_loss/num_batch), \
                            ' posls ', "{0:.3f}".format(total_pos_ls/num_batch), 'predls', "{0:.3f}".format(total_pred_ls/num_batch), \
                            'distls', "{0:.3f}".format(total_dist_ls/num_batch),
                            ' posxyzls', "{0:.3f}".format(total_posxyz_ls/num_batch),
                            ' explore', "{0:.3f}".format(2-coll_acc-off_accuracy))
                        with open(args.save_path+'/log_train.txt', 'a') as fi:
                            fi.write('collacc '+'{0:.3f}'.format(coll_acc)+' offacc '+'{0:.3f}'.format(off_accuracy)+\
                                    ' distls '+'{0:.3f}'.format(total_dist_ls/num_batch)+'\n')
                    else:
                        print('collacc', "{0:.3f}".format(coll_acc), \
                            "{0:.3f}".format(total_coll_ls/num_batch), \
                            'offacc', "{0:.3f}".format(off_accuracy), \
                            "{0:.3f}".format(total_off_ls/num_batch), \
                            'spls', "{0:.3f}".format(speed_loss/num_batch), 'ttls', "{0:.3f}".format(total_loss/num_batch), \
                            ' posls ', "{0:.3f}".format(total_pos_ls/num_batch), 'predls', "{0:.3f}".format(total_pred_ls/num_batch), \
                            'distls', "{0:.3f}".format(total_dist_ls/num_batch),
                            ' posxyzls', "{0:.3f}".format(total_posxyz_ls/num_batch),
                            ' explore', "{0:.3f}".format(2-coll_acc-off_accuracy))
                    #if total_dist_ls/num_batch >= 14.0:
                    #    for param_group in optimizer.param_groups:
                    #        param_group['lr'] = 0.0001
                except:
                    print('dist ls', total_dist_ls/num_batch)
                try: # ratio 1 2 4 8 16
                    coll_acc_0 = (cnf_matrix[1,0]+cnf_matrix[1,1]+1)/(cnf_matrix.sum()*1.0)
                    coll_acc_1 = (cnf_matrix[0,0]+cnf_matrix[0,1]+1)/(cnf_matrix.sum()*1.0)
                    off_acc_0 = (cnf_matrix_off[1,0]+cnf_matrix_off[1,1]+1)/(cnf_matrix_off.sum()*1.0)
                    off_acc_1 = (cnf_matrix_off[0,0]+cnf_matrix_off[0,1]+1)/(cnf_matrix_off.sum()*1.0)
                except:
                    pass
                pkl.dump(cnf_matrix, open(args.save_path+'/cnfmat/'+str(epoch)+'.pkl', 'wb'))
                pkl.dump(cnf_matrix_off, open(args.save_path+'/cnfmat/off_'+str(epoch)+'.pkl','wb')) 
                epoch+=1
                num_epoch += 1
                if num_epoch >= 10:# or total_dist_ls/num_batch<=5.0:
                    sign = False
                if epoch % save_freq == 0:
                    torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(0).zfill(9)+'.pt')
                    torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optim_'+str(0).zfill(9)+'.pt')
    return net 
