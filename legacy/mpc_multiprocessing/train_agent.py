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

def train_agent(args, memory_pool, worker_id = 0):
    logger = setup_logger('worker_%d' % worker_id, os.path.join(args.save_path, 'worker_log_%d.txt' % worker_id))

    train_net = ConvLSTMMulti(3, 3, args.num_total_act, True, \
                multi_info = False, \
                with_posinfo = args.with_posinfo, \
                use_pos_class = args.use_pos_class, \
                with_speed = args.with_speed, \
                with_pos = args.with_pos, \
                frame_history_len = args.frame_history_len,
                with_dla = args.with_dla)
    if torch.cuda.is_available():
        train_net = train_net.cuda()

    params = [param for param in train_net.parameters() if param.requires_grad]
    optimizer = optim.Adam(params, lr = args.lr)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    train_net, epoch = load_model(args.save_path+'/model', train_net, data_parallel = True)
    train_net.train()
    return

    while not memory_pool.can_sample(args.batch_size):
        time.sleep(1)

    for tt in range(int(args.num_steps // args.learning_freq)):
        # start training
        if memory_pool.can_sample(args.batch_size):
            print('Start training.') 
            for num_epoch in range(10):
                num_batch, all_label, all_pred, off_label, off_pred, prev_loss, speed_loss = 0, [], [], [], [], 10000.0, 0
                total_loss = 0
                total_coll_ls = 0
                total_off_ls = 0
                total_pred_ls = 0
                total_pos_ls = 0
                total_dist_ls = 0
                total_posxyz_ls = 0
                
                x, idxes = memory_pool.sample(batch_size)
                x = list(x)

                act_batch     = Variable(torch.from_numpy(x[0]), requires_grad=False).type(dtype)
                coll_batch    = Variable(torch.from_numpy(x[1]), requires_grad=False).type(dtype)
                speed_batch   = Variable(torch.from_numpy(x[2]), requires_grad=False).type(dtype)
                offroad_batch = Variable(torch.from_numpy(x[3]), requires_grad=False).type(dtype)
                dist_batch    = Variable(torch.from_numpy(x[4]), requires_grad=False).type(dtype)
                img_batch     = Variable(torch.from_numpy(x[5])[:, 0, :, :, :], requires_grad=False).type(dtype)
                nximg_batch   = Variable(torch.from_numpy(x[6])[:, 0, :, :, :], requires_grad=False).type(dtype)
                pos_batch     = Variable(torch.from_numpy(x[7])[:, 0, :], requires_grad=False).type(dtype)
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
                if args.with_speed:
                    speed_ls = nn.L1Loss()(pred_sp, speed_batch[:,1:,:])
                    speed_loss += float(speed_ls.data.cpu().numpy())
                else:
                    speed_ls = 0
                dist_ls = nn.MSELoss(reduce=False)(pred_dist.view(-1,pred_step), dist_batch[:,1:].view(-1,pred_step))
                dist_ls = dist_ls.view(-1)
                if args.use_pos_class == False and args.with_pos:
                    this_weight = weight.repeat(int(pos_batch.size()[0]), 1,1)
                    pred_pos = pred_pos * this_weight
                    pos_batch = pos_batch[:,1:,:] * this_weight
                    pos_ls = nn.L1Loss()(pred_pos, pos_batch)
                elif args.with_pos:
                    pos_batch = pos_batch[:,1:,:].contiguous()
                    pos_ls = Focal_Loss(pred_pos.view(-1,19), (torch.max(pos_batch.view(-1,19),-1)[1]).view(-1))
                pred_ls = nn.L1Loss(reduce=False)(pred_enc, nximg_enc).sum(-1).view(-1)
                loss = pred_ls + coll_ls + offroad_ls + dist_ls# + 20*pos_ls 
                loss_np = loss.data.cpu().numpy().reshape((-1, args.pred_step)).sum(-1)
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
                    print('dist ls', total_dist_ls / num_batch)
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
                if epoch % save_freq == 0:
                    torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(0).zfill(9)+'.pt')
                    torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optim_'+str(0).zfill(9)+'.pt')