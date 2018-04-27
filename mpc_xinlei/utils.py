import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import os
import PIL.Image as Image
import random
from sklearn.metrics import confusion_matrix
import pdb

class MPCData(Dataset):
    def __init__(self, mpc_buffer, batch_size=1):
        self.mpc_buffer = mpc_buffer
        self.batch_size = batch_size

    def update_buffer(self, mpc_buffer):
        self.mpc_buffer = mpc_buffer

    def __len__(self):
        return self.mpc_buffer.num_in_buffer-2

    def __getitem__(self, idx):
        done = self.mpc_buffer.sample_done(idx)
        while done == False:
            idx = random.randint(0, self.mpc_buffer.num_in_buffer-2)
            done = self.mpc_buffer.sample_done(idx) 
        x = list(self.mpc_buffer._encode_sample([idx]))
        act_batch, coll_batch, offroad_batch, dist_batch, img_batch, nximg_batch = x[0], x[1], x[3], x[4], x[5], x[6]
        return act_batch.squeeze(), coll_batch.squeeze(), offroad_batch.squeeze(), dist_batch.squeeze(0), img_batch.squeeze(), nximg_batch.squeeze()

def train_model(train_net, mpc_buffer, batch_size, epoch, avg_img_t, std_img_t, pred_step=15):
    if epoch % 20 == 0:
        x, idxes = mpc_buffer.sample(batch_size, sample_early = True)
    else:
        x, idxes = mpc_buffer.sample(batch_size, sample_early = False)
    x = list(x)
    for iii in range(len(x)):
        x[iii] = torch.from_numpy(x[iii]).float().cuda()
    act_batch = Variable(x[0], requires_grad=False)
    coll_batch = Variable(x[1], requires_grad=False)
    offroad_batch = Variable(x[3], requires_grad=False)
    dist_batch = Variable(x[4])
    img_batch = Variable(((x[5].float()-avg_img_t)/(std_img_t+0.0001)), requires_grad=False)
    nximg_batch = Variable(((x[6].float()-avg_img_t)/(std_img_t+0.0001)), requires_grad=False)
    with torch.no_grad():
        nximg_enc = train_net(nximg_batch, get_feature=True)
        nximg_enc = nximg_enc.detach()
    pred_coll, pred_enc, pred_off, pred_dist,_,_ = train_net(img_batch, act_batch, int(act_batch.size()[1]))
    coll_ls = Focal_Loss(pred_coll.view(-1,2), (torch.max(coll_batch.view(-1,2),-1)[1]).view(-1), reduce=True)
    offroad_ls = Focal_Loss(pred_off.view(-1,2), (torch.max(offroad_batch.view(-1,2),-1)[1]).view(-1), reduce=True)
    dist_ls = torch.sqrt(nn.MSELoss()(pred_dist.view(-1,pred_step), dist_batch[:,1:].view(-1,pred_step)))
    print('prediction', pred_dist[0,:].data.cpu().numpy(), 'gr', dist_batch[0,:].data.cpu().numpy())
    pred_ls = nn.L1Loss()(pred_enc, nximg_enc).sum()
    loss = pred_ls + coll_ls + offroad_ls + 10*dist_ls
    coll_acc, off_acc, total_dist_ls = log_info(pred_coll, coll_batch, pred_off, offroad_batch, \
        float(coll_ls.data.cpu().numpy()), float(offroad_ls.data.cpu().numpy()),\
        float(pred_ls.data.cpu().numpy()), float(dist_ls.data.cpu().numpy()), \
        float(loss.data.cpu().numpy()), epoch, 1)
    return loss, coll_acc, off_acc, total_dist_ls
 
class DoneCondition:
    def __init__(self, size):
        self.size = size
        self.off_cnt = 0
        self.pos = []

    def isdone(self, pos, dist, posxyz):
        if pos <=-6.2 and dist < 0:
            self.off_cnt += 1
        elif pos > -6.2 or dist > 0:
            self.off_cnt = 0
        if self.off_cnt > self.size:
            return True
        if abs(pos) >= 21.0:
            return True
        self.pos.append(list(posxyz))
        real_pos = np.concatenate(self.pos[-50:])
        real_pos = real_pos.reshape(-1,3)
        std = np.sum(np.std(real_pos, 0))
        if std < 3.0 and len(self.pos) > 50:
            self.pos = []
            return True
        return False 

class ObsBuffer:
    def __init__(self, frame_history_len=3):
        self.frame_history_len = frame_history_len
        self.last_obs_all = []

    def store_frame(self, frame, avg_img, std_img):
        obs_np = (frame-avg_img)/(std_img+0.0001)
        obs_np = obs_np.transpose(2,0,1)
        if len(self.last_obs_all) < self.frame_history_len:
            self.last_obs_all = []
            for ii in range(self.frame_history_len):
                self.last_obs_all.append(obs_np)
        else:
            self.last_obs_all = self.last_obs_all[1:] + [obs_np]
        return np.concatenate(self.last_obs_all, 0)

    def clear(self):
        self.last_obs_all = []
        return

def log_info(pred_coll, coll_batch, pred_off, offroad_batch, \
            total_coll_ls, total_off_ls, total_pred_ls, total_dist_ls, total_loss, \
            epoch, num_batch):
    coll_label, coll_pred, off_label, off_pred = [], [], [], []
    pred_coll_np = pred_coll.view(-1,2).data.cpu().numpy()
    coll_np = coll_batch.view(-1,2).data.cpu().numpy()
    pred_coll_np = np.argmax(pred_coll_np, 1)
    coll_np = np.argmax(coll_np, 1)
    coll_label.append(coll_np)
    coll_pred.append(pred_coll_np)

    pred_off_np = pred_off.view(-1,2).data.cpu().numpy()
    off_np = offroad_batch.view(-1,2).data.cpu().numpy()
    pred_off_np = np.argmax(pred_off_np, 1)
    off_np = np.argmax(off_np, 1)
    off_label.append(off_np)
    off_pred.append(pred_off_np)
    
    coll_label = np.concatenate(coll_label)
    coll_pred = np.concatenate(coll_pred)
    off_label = np.concatenate(off_label)
    off_pred = np.concatenate(off_pred)
    cnf_matrix = confusion_matrix(coll_label, coll_pred)
    cnf_matrix_off = confusion_matrix(off_label, off_pred)
    coll_acc, off_accuracy = 0, 0
    try:
        coll_acc = (cnf_matrix[0,0]+cnf_matrix[1,1])/(cnf_matrix.sum()*1.0)
        off_accuracy = (cnf_matrix_off[0,0]+cnf_matrix_off[1,1])/(cnf_matrix_off.sum()*1.0)
        if epoch % 20 == 0:
            print('sample early collacc', "{0:.3f}".format(coll_acc), \
                "{0:.3f}".format(total_coll_ls/num_batch), \
                'offacc', "{0:.3f}".format(off_accuracy), \
                "{0:.3f}".format(total_off_ls/num_batch), \
                'ttls', "{0:.3f}".format(total_loss/num_batch), \
                 'predls', "{0:.3f}".format(total_pred_ls/num_batch), \
                'distls', "{0:.3f}".format(total_dist_ls/num_batch))
        else:
            print('collacc', "{0:.3f}".format(coll_acc), \
                "{0:.3f}".format(total_coll_ls/num_batch), \
                'offacc', "{0:.3f}".format(off_accuracy), \
                "{0:.3f}".format(total_off_ls/num_batch), \
                'ttls', "{0:.3f}".format(total_loss/num_batch), \
                'predls', "{0:.3f}".format(total_pred_ls/num_batch), \
                'distls', "{0:.3f}".format(total_dist_ls/num_batch))
    except:
        print('dist ls', total_dist_ls/num_batch)
    return coll_acc, off_accuracy, total_dist_ls/num_batch

def init_dirs(dir_list):
    for path in dir_list:
        make_dir(path)

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def load_model(path, net, data_parallel = True, optimizer = None, resume=True):
    if resume:
        file_list = sorted(os.listdir(path+'/model')) 
        model_path = file_list[-1]
        epoch = pkl.load(open(path+'/epoch.pkl', 'rb'))
        state_dict = torch.load(os.path.join(path, 'model', model_path))
        net.load_state_dict(state_dict)
        print('load success')
    else:
        epoch = 0
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net) if data_parallel else net
        net = net.cuda()

    if optimizer is not None and epoch > 0:
        optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer/optim_' + model_path.split('_')[-1])))

    if optimizer is None:
        return net, epoch
    else:
        return net, epoch, optimizer
        
def get_info_np(info, use_pos_class = False):
    speed_np = np.array([[info['speed'], info['angle']]])
    if use_pos_class:
        pos = int(round(np.clip(info['trackPos'], -9, 9)) + 9)
        pos_np = np.zeros((1, 19))
        pos_np[0, pos] = 1
    else:
        pos_np = np.array([[info['trackPos']]])
    posxyz_np = np.array([list(info['pos'])])
    return speed_np, pos_np, posxyz_np

def get_info_ls(info):
    speed = [info['speed'], info['angle']]
    pos = [info['trackPos']] + list(info['pos'])
    return speed, pos

def action_sampler(num_step=30, prev_act=1):
    '''
    action meanings: 0: turn left, accelerate
                    1: accelerate
                    2: turn right, accelerate
                    3: turn left
                    4: do nothing
                    5: turn right
                    6: turn left, decelerate
                    7: decelerate
                    8: turn right, decelerate
    '''
    actions = []
    current_act = prev_act
    for i in range(num_step):
        if current_act == 1 or current_act == 4:
            p = np.ones(9)*(1-0.9)/7.0
            p[1] = 0.45
            p[4] = 0.45
        elif current_act in [0,2]:
            p = np.ones(9)*(1-0.6)/7.0
            p[0] = 0.3
            p[2] = 0.3
        elif current_act in [3,5]:
            p = np.ones(9)*(1-0.6)/7.0
            p[3] = 0.3
            p[5] = 0.3
        else:
            p = np.ones(9)*(1-0.3)/8.0
            p[current_act] = 0.3
        current_act = np.random.choice(9,p=list(p/p.sum()))
        actions.append(current_act)
    return actions

def offroad_loss(probs, target, reduce=True):
    target = torch.abs(target)
    target = torch.clamp(target, min=0, max=8)
    target = target/8.0
    target = ((target>=0.5).float()*target+0.1)*(target>=0.5).float()+((target<0.5).float()*target-0.1)*(target<0.5).float()
    target = torch.clamp(target, min=0.001, max=1)
    target = target.repeat(1,2)
    target[:,0] = 1-target[:,0]
    # loss = nn.MSELoss(reduce=reduce)(probs, target)
    loss = (-1.0*target*torch.log(probs)).sum(-1)
    if reduce:
        loss = loss.sum()/(loss.size()[0])
    return loss

def Focal_Loss(probs, target, reduce=True):
    # probs : batch * num_class
    # target : batch,
    loss = -1.0 * (1-probs).pow(1) * torch.log(probs)
    batch_size = int(probs.size()[0])
    loss = loss[torch.arange(batch_size).long().cuda(), target.long()]
    if reduce == True:
        loss = loss.sum()/(batch_size*1.0)
    return loss

def Focal_Loss_Regress(probs, target, reduce=True):
    # probs: batch * num_class
    # target: batch * num_class
    target_class = (torch.max(target, -1)[1]).view(-1,1)
    target = target * 2 - 1
    res1 = -1.0*(torch.log(probs) * target)
    weight = Variable(torch.arange(19)).view(1,19).repeat(probs.size()[0], 1).cuda().float()
    weight = 0.1*torch.abs(weight - target_class.repeat(1, 19).float())+1
    loss = (weight * res1).sum(-1)
    if reduce == True:
        loss = loss.sum()/(probs.size()[0]*19.0)
    return loss

class PredData(Dataset):
    def __init__(self, data_dir, act_dir, done_dir, coll_dir, 
                speed_dir, offroad_dir, pos_dir, sample_len, num_actions, use_pos_class=False, frame_history_len=4):
        self.data_dir = data_dir
        self.act_dir = act_dir
        self.done_dir = done_dir
        self.coll_dir = coll_dir
        self.speed_dir = speed_dir
        self.offroad_dir = offroad_dir
        self.pos_dir = pos_dir
        self.data_files = sorted(os.listdir(data_dir))
        self.act_files = sorted(os.listdir(act_dir))
        self.done_files = sorted(os.listdir(done_dir))
        self.coll_files = sorted(os.listdir(coll_dir))
        self.speed_files = sorted(os.listdir(speed_dir))
        self.offroad_files = sorted(os.listdir(offroad_dir))
        self.pos_files = sorted(os.listdir(pos_dir))
        self.length = min(len(self.data_files), len(self.act_files))
        self.sample_len = sample_len
        self.num_actions = num_actions
        self.use_pos_class = use_pos_class
        self.frame_history_len = frame_history_len
    
    def __len__(self):
        return self.length

    def sample_done(self, idx, sample_len):
        if idx < 10 or idx >= self.length - sample_len -1 :
            return False
        else:
            done_list = []
            create_times = []
            for i in range(sample_len):
                try:
                    done_list.append(1.0*pkl.load(open(os.path.join(self.done_dir,str(idx-self.frame_history_len).zfill(9)+'.pkl'), 'rb')))
                    create_times.append(os.path.getmtime(os.path.join(self.done_dir, str(idx-self.frame_history_len).zfill(9)+'.pkl')))
                except:
                    return False
                idx += 1
            if np.sum(done_list) >= 1.0:
                return False
            else:
                create_times = np.array(create_times)
                create_time = np.abs(create_times[1:]-create_times[:-1])
                if np.any(create_time > 100):
                    return False
                else:
                    return True

    def reinit(self):
        self.data_files = sorted(os.listdir(self.data_dir))
        self.act_files = sorted(os.listdir(self.act_dir))
        self.done_files = sorted(os.listdir(self.done_dir))
        self.coll_files = sorted(os.listdir(self.coll_dir))
        self.speed_files = sorted(os.listdir(self.speed_dir))
        self.offroad_files = sorted(os.listdir(self.offroad_dir))
        self.pos_files = sorted(os.listdir(self.pos_dir))
        self.length = min(len(self.data_files), len(self.act_files))

    def __getitem__(self, idx):
        self.reinit()
        sign_coll = False
        sign_off = False
        sample_time = 0
        if random.random() <= 0.45:
            should_sample = True
        else:
            should_sample = False
        while sign_coll == False or sign_off == False:
            if should_sample == True:
                sign_coll = True
                sign_off = True 
            can_sample = self.sample_done(idx, self.sample_len+self.frame_history_len)
            while can_sample == False:
                idx = np.random.randint(0, self.length-self.sample_len, 1)[0]
                can_sample = self.sample_done(idx, self.sample_len+self.frame_history_len)
            act_list = []
            coll_list = []
            speed_list = []
            offroad_list = []
            dist_list = []
            pos_list = []
            posxyz_list = []
            dist = 0.0
            sample_time +=1
            img_final = np.zeros((self.sample_len, 3*self.frame_history_len, 256, 256))
            nximg_final = np.zeros((self.sample_len, 3*self.frame_history_len, 256, 256))
            prev_pos = 0.0
            try:
                for i in range(self.sample_len):
                    action = pkl.load(open(os.path.join(self.act_dir, str(idx).zfill(9)+'.pkl'), 'rb'))
                    try: 
                        action = action[0]
                    except:
                        pass
                    speed = pkl.load(open(os.path.join(self.speed_dir, str(idx).zfill(9)+'.pkl'), 'rb'))
                    speed_list.append(speed)
                    offroad = pkl.load(open(os.path.join(self.offroad_dir, str(idx).zfill(9)+'.pkl'),'rb'))
                    off = np.zeros(2)
                    off[int(offroad)] = 1.0
                    offroad_list.append(off)
                    pos = pkl.load(open(os.path.join(self.pos_dir, str(idx).zfill(9)+'.pkl'), 'rb'))
                    if i == 0:
                        pos_list.append(pos[0])
                        prev_pos = pos[0]
                    else:
                        # pos_list.append(pos[0]-prev_pos)
                        # prev_pos = pos[0]
                        pos_list.append(pos[0])
                    posxyz_list.append(np.array(pos[1:]))
                    dist = speed[0]*(np.cos(speed[1])-np.abs(np.sin(speed[1]))-((np.abs(pos[0]-2))/9.0)**2.0)
                    if self.use_pos_class == False:
                        dist = speed[0]*(np.cos(speed[1])-np.abs(np.sin(speed[1]))-((np.abs(pos[0]-2))/9.0)**2.0)
                    dist_list.append(dist)
                    act = np.zeros(self.num_actions)
                    act[int(action)] = 1.0
                    # act = np.exp(act)/np.exp(act).sum()
                    act_list.append(act)
                    coll = np.zeros(2)
                    collision = pkl.load(open(os.path.join(self.coll_dir, str(idx).zfill(9)+'.pkl'), 'rb'))
                    if collision == 1:
                        sign_coll= True
                    sign_off = True 
                    coll[int(collision)] = 1.0
                    coll_list.append(coll)
                    this_imgs = []
                    this_nx_imgs = []
                    for ii in range(self.frame_history_len):
                        this_img = np.array(Image.open(os.path.join(self.data_dir, str(idx-self.frame_history_len+1+ii).zfill(9)+'.png')))
                        this_nx_img = np.array(Image.open(os.path.join(self.data_dir, str(idx-self.frame_history_len+2+ii).zfill(9)+'.png')))
                        this_img = this_img.transpose(2,0,1)
                        this_nx_img = this_nx_img.transpose(2,0,1)
                        this_imgs.append(this_img)
                        this_nx_imgs.append(this_nx_img)
                    img_final[i,:] = np.concatenate(this_imgs)
                    nximg_final[i,:] = np.concatenate(this_nx_imgs)
                    idx += 1
                    if sample_time == 20:
                        sign_coll = True
                        sign_off = True 
                speed = pkl.load(open(os.path.join(self.speed_dir, str(idx).zfill(9)+'.pkl'), 'rb'))
                speed_list.append(speed)
                pos = pkl.load(open(os.path.join(self.pos_dir, str(idx).zfill(9)+'.pkl'), 'rb'))
                pos_list.append(pos[0])
                posxyz_list.append(np.array(pos[1:]))
                dist = speed[0] * (np.cos(speed[1])-np.abs(np.sin(speed[1]))-(np.abs(pos[0]-2.0)/9.0)**2.0)
                dist_list.append(dist)
            except:
                sign_coll = False
                sign_off = False
        if self.use_pos_class:
            return np.stack(act_list), np.stack(coll_list), np.stack(speed_list),np.stack(offroad_list), np.stack(dist_list), img_final, nximg_final, np.stack(pos_list).reshape((-1,19)), np.stack(posxyz_list)
        else:
            return np.stack(act_list),np.stack(coll_list),np.stack(speed_list),np.stack(offroad_list),np.stack(dist_list),img_final,nximg_final,np.stack(pos_list).reshape((-1,1)), np.stack(posxyz_list)
