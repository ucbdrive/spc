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
from sklearn.metrics import confusion_matrix
from train_policy import TrainData

def make_dir(path):
    if os.path.isdir(path) == False:
        print('make ', path)
        os.mkdir(path)
    return

def load_model(path, net, use_cuda=True, data_parallel=True):
    file_list = os.listdir(path)
    file_list = sorted(file_list)
    model_path = file_list[-2]
    state_dict = torch.load('model/'+model_path)
    net.load_state_dict(state_dict)
    if use_cuda == True and data_parallel == False:
        net = net.cuda()
    if data_parallel == True and use_cuda == True:
        net = torch.nn.DataParallel(net).cuda()
    epoch = int(model_path.split('_')[2].split('.')[0])
    print('load model', model_path)
    return net, epoch

def test_model(normalize=True, pred_step=15, batch_size=1):
    use_pos_class = False#True
    net = ConvLSTMMulti(3,3,6, True, multi_info=False, with_posinfo=False, use_pos_class=use_pos_class, with_speed=False,frame_history_len=3)
    net.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    avg_img = pkl.load(open('avg_img.pkl','rb'))
    std_img = pkl.load(open('std_img.pkl','rb'))
    avg_img_t = torch.from_numpy(avg_img.transpose(2,0,1)).float().repeat(3,1,1)
    std_img_t = torch.from_numpy(std_img.transpose(2,0,1)).float().repeat(3,1,1)
    net, epoch = load_model('model', net, use_cuda=True, data_parallel=False)
    for param in net.parameters():
        param.requires_grad = False

    traindatafast = TrainData('data_fast', num_time=pred_step, buffer_size=1000, use_pos_class=use_pos_class)
    traindataslow = TrainData('data_slow', num_time=pred_step, buffer_size=50000, use_pos_class=use_pos_class)
    datafast = PredData(traindatafast.data_dir, traindatafast.action_dir, traindatafast.done_dir, traindatafast.coll_dir, traindatafast.speed_dir, traindatafast.offroad_dir, traindatafast.pos_dir, pred_step, 6, use_pos_class=use_pos_class, frame_history_len=3)
    dataslow = PredData(traindataslow.data_dir, traindataslow.action_dir, traindataslow.done_dir, traindataslow.coll_dir, traindataslow.speed_dir, traindataslow.offroad_dir, traindataslow.pos_dir, pred_step, 6, use_pos_class=use_pos_class, frame_history_len=3)
    loaderfast = DataLoader(dataset=datafast, batch_size=batch_size, num_workers=16, shuffle=True)
    loaderslow = DataLoader(dataset=dataslow, batch_size=batch_size, num_workers=16, shuffle=True)
    sign = True
    while sign:
        num_batch, all_label, all_pred, off_label, off_pred, prev_loss, speed_loss = 0, [], [], [], [], 10000.0, 0
        total_loss = 0
        total_coll_ls = 0
        total_off_ls = 0
        total_pred_ls = 0
        total_pos_ls = 0
        total_dist_ls = 0
        datafast.reinit()
        dataslow.reinit()
        if random.random() < 0.3:
            this_loader = loaderslow
        else:
            this_loader = loaderfast
        total_pos_ls = 0
        for x in this_loader:
            act_batch = Variable(x[0], requires_grad=False).type(dtype)
            coll_batch = Variable(x[1], requires_grad=False).type(dtype)
            speed_batch = Variable(x[2], requires_grad=False).type(dtype)
            offroad_batch = Variable(x[3], requires_grad=False).type(dtype)
            dist_batch = Variable(x[4]).type(dtype)
            if normalize:
                img_batch = Variable(((x[5].float()-avg_img_t)/(std_img_t+0.0001)).type(dtype))
                nximg_batch = Variable(((x[6].float()-avg_img_t)/(std_img_t+0.0001)).type(dtype))
            else:
                img_batch = Variable(x[5], requires_grad=False).type(dtype)/255.0
                nximg_batch = Variable(x[6], requires_grad=False).type(dtype)/255.0
            pos_batch = Variable(x[7]).type(dtype)
            posxyz_batch = Variable(x[8]).type(dtype)
            pred_coll, pred_enc, pred_off, pred_sp, pred_dist,pred_pos,pred_posxyz,_,_ = net(img_batch, act_batch, speed_batch, pos_batch, int(act_batch.size()[1]),posxyz=posxyz_batch)
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
            pdb.set_trace()
            # total_pos_ls += nn.L1Loss()(pred_pos, pos_batch[:,1:,:])
            # pos_batch = pos_batch[:,1:,:].contiguous()
            # pos_ls = Focal_Loss(pred_pos.view(-1,19), (torch.max(pos_batch.view(-1,19),-1)[1]).view(-1))
            # total_pos_ls += pos_ls
            num_batch += 1
            if num_batch == 1:
                print('action is ', torch.max(act_batch.view(-1,9),-1)[1].data.cpu().numpy())
                #if use_pos_class:
                #    print('this pos', (torch.max(pos_batch.view(-1,19),-1)[1]-9).data.cpu().numpy().reshape((-1)), 'pred', (torch.max(pred_pos.view(-1,19),-1)[1]-9).data.cpu().numpy().reshape((-1)))
                #else:
                #    print('this pos', pos_batch[:,1:,:].data.cpu().numpy().reshape((-1)), pred_pos.data.cpu().numpy().reshape((-1)))
                #print('pred off', pred_off.data.cpu().numpy().reshape((-1,2)))
                #print('gt off', off_np)
                #print('pred dist', pred_dist.data.cpu().numpy().reshape((-1)))
                #print('gt dist', dist_batch.data.cpu().numpy().reshape((-1)))
            if num_batch >= 5:
                break
        all_label = np.concatenate(all_label)
        all_pred = np.concatenate(all_pred)
        off_label = np.concatenate(off_label)
        off_pred = np.concatenate(off_pred)
        cnf_matrix = confusion_matrix(all_label, all_pred)
        cnf_matrix_off = confusion_matrix(off_label, off_pred)
        # total_pos_ls /= (num_batch)
        try:
            coll_acc = (cnf_matrix[0,0]+cnf_matrix[1,1])/(cnf_matrix.sum()*1.0)
            off_accuracy = (cnf_matrix_off[0,0]+cnf_matrix_off[1,1])/(cnf_matrix_off.sum()*1.0)
            print('collacc', coll_acc, 'offacc', off_accuracy)#, 'pos ls', total_pos_ls.data.cpu().numpy())
        except:
            pass #print('pos los', total_pos_ls.data.cpu().numpy())
        net, epoch = load_model('model', net, use_cuda=True, data_parallel=False)
    return net

test_model() 
