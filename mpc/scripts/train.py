import torch
import torch.nn as nn
from model import *
from utils import *
import torch.optim as optim
import pdb
from sklearn.metrics import confusion_matrix
import pickle as pkl
import os
import cv2
from collections import OrderedDict

cv2.setNumThreads(0)

def train(epoch, dataset, data_loader, net, num_actions, use_cuda, start_epoch=0):
    if use_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    net = net.type(dtype)
    net.train()
    train_loss = 0
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for ep in range(start_epoch, start_epoch+epoch):
        train_loss = 0.0
        coll_loss = 0.0
        t = 0
        acc = []
        all_label = []
        all_pred = []
        off_label = []
        off_pred = []
        off_accs = []
        offroad_loss = 0.0
        speed_loss = 0.0
        dist_loss = 0.0
        data_len = len(os.listdir('data/data'))
        for x in data_loader:
            t += 1
            act = Variable(x[0]).type(dtype)
            coll = Variable(x[1]).type(dtype)
            speed = Variable(x[2]).type(dtype)
            offroad = Variable(x[3]).type(dtype)
            dist = Variable(x[4]).type(dtype)
            img = Variable(x[5]).type(dtype)
            nximg = Variable(x[6]).type(dtype)
            optimizer.zero_grad()
            pred_coll, pred_enc, pred_off, pred_sp, pred_dist = net(img, act, speed, act.size()[1])
            nximg_enc = net(nximg, get_feature=True).detach()
            
            # collision loss
            # factor = coll.view(-1,2).max(1)[1]
            # factor = torch.stack([factor]*2).transpose(1,0).float()
            # if use_cuda:
            #    factor = factor.cuda()
            # coll_ls = (((pred_coll.view(-1,2)-coll.view(-1,2))**2)*(1+(factor)*10)).sum()/pred_coll.view(-1,2).size()[0]
            coll_ls = nn.CrossEntropyLoss()(pred_coll.view(-1,2), torch.max(coll.view(-1,2),-1)[1])
            # offroad loss
            #try:
            #    factor2 = offroad.view(-1, 2).max(1)[1]
            #except:
            #    pdb.set_trace()
            #factor2 = torch.stack([factor2]*2).transpose(1,0).type(dtype)
            #offroad_ls = (((pred_off.view(-1,2)-offroad.view(-1,2))**2)*(1+factor2*10)).sum()/pred_off.view(-1,2).size()[0]
            offroad_ls = nn.CrossEntropyLoss()(pred_off.view(-1,2), torch.max(offroad.view(-1,2), -1)[1])

            # speed loss
            # speed_ls = (((pred_sp.view(-1,2)-speed.view(-1,2))**2)).sum()/pred_sp.view(-1,2).size()[0]
            speed_ls = nn.MSELoss()(pred_sp.view(-1,2), speed.view(-1,2))

            # dist loss
            # dist_ls = (((pred_dist.view(-1,1)-dist.view(-1,1))**2)).sum()/pred_sp.view(-1,1).size()[0]
            dist_ls = nn.MSELoss()(pred_dist.view(-1,1), dist.view(-1,1))

            this_acc = (pred_coll.view(-1,2).max(1)[1] == coll.view(-1, 2).max(1)[1]).sum()
            this_acc = float(this_acc.data.cpu().numpy())
            this_acc /= int(pred_coll.view(-1,2).size()[0])*1.0

            # offroad acc
            off_acc = (pred_off.view(-1,2).max(1)[1] == offroad.view(-1,2).max(1)[1]).sum()
            off_acc = float(off_acc.data.cpu().numpy())
            off_acc /= int(pred_off.view(-1,2).size()[0])*1.0

            # confusion matrix
            pred_coll_np = pred_coll.view(-1,2).data.cpu().numpy()
            coll_np = coll.view(-1,2).data.cpu().numpy()
            pred_coll_np = np.argmax(pred_coll_np, 1)
            coll_np = np.argmax(coll_np, 1)
            all_label.append(coll_np)
            all_pred.append(pred_coll_np)

            pred_off_np = pred_off.view(-1,2).data.cpu().numpy()
            off_np = offroad.view(-1,2).data.cpu().numpy()
            pred_off_np = np.argmax(pred_off_np, 1)
            off_np = np.argmax(off_np, 1)
            off_label.append(off_np)
            off_pred.append(pred_off_np)
            acc.append(this_acc) #float(this_acc.data.cpu().numpy()))
            off_accs.append(off_acc)

            loss = 0.001 * nn.MSELoss()(pred_enc, nximg_enc)+coll_ls + offroad_ls + speed_ls + dist_ls
            loss.backward()
            train_loss += loss.data[0]
            coll_loss += coll_ls.data[0]
            offroad_loss += offroad_ls.data[0]
            speed_loss += speed_ls.data[0]
            dist_loss += dist_ls.data[0]
            optimizer.step()
            if t >= 2:
                break
        all_label = np.concatenate(all_label)
        all_pred = np.concatenate(all_pred)
        off_label = np.concatenate(off_label)
        off_pred = np.concatenate(off_pred)
        cnf_matrix = confusion_matrix(all_label, all_pred)
        cnf_matrix_off = confusion_matrix(off_label, all_pred)
        pkl.dump(cnf_matrix, open('cnfmat/'+str(ep)+'.pkl', 'wb'))
        pkl.dump(cnf_matrix_off, open('cnfmat/off_'+str(ep)+'.pkl','wb'))
        print('epoch ', ep, ' loss ', train_loss, ' coll loss is', coll_loss, 'acc is', np.mean(acc), \
            'off loss ', offroad_loss, ' acc is ', np.mean(off_accs), 'speed loss ', speed_loss, 'dist loss', dist_loss)
        with open('log_loss.txt', 'a') as fi:
            fi.write('epoch '+str(ep)+' loss '+str(train_loss)+' coll loss is '+str(coll_loss)+\
                        'acc is'+str(np.mean(acc))+\
                    ' offloss is '+str(offroad_loss)+\
                    ' acc '+str(np.mean(off_accs))+\
                    ' speed loss '+str(speed_loss)+\
                    ' dist loss '+str(dist_loss)+'.\n')
        torch.save(net.state_dict(), 'model/pred_model_'+str(ep).zfill(5)+'.pt') 
    return net

use_cuda = torch.cuda.is_available()
net = ConvLSTMMulti(3, 3, 9, True)
models = sorted(os.listdir('model'))
try:
    model_path = 'model/'+models[-1]
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    start_ep = int(models[-1].split('_')[2].split('.')[0])
except:
    start_ep = 0
net = torch.nn.DataParallel(net).cuda()
epoch = 2000000
data = PredData('data/data', 'data/action', 'data/done', 'data/coll', 'data/speed', 'data/offroad', 7, 9)
loader = DataLoader(dataset=data, batch_size=14, num_workers=8, shuffle=True)
_ = train(epoch, data, loader, net, 9,  use_cuda, start_epoch=start_ep) 
