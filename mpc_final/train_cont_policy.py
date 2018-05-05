from model import *
import torch
import gym

def train_policy(args, 
                 env,
                 num_steps=40000000,
                 batch_size=32,
                 pred_step=15,
                 normalize=True,
                 buffer_size=50000,
                 save_path='model',
                 save_freq=10,
                 frame_history_len=3,
                 num_total_act=2,
                 use_seg=True,
                 use_xyz=True,
                 use_dqn=True):
    ''' create model '''
    train_net = ConvLSTMMulti(num_total_act, True, frame_history_len, use_seg, use_xyz, 'drn_d_22', 4, 1024, 32).train()
    net = ConvLSTMMulti(num_total_act, True, frame_history_len, use_seg, use_xyz, 'drn_d_22', 4, 1024, 32).eval()
    if use_dqn:
        dqn_net = atari_model(3 * frame_history_len, 11, frame_history_len).cuda().float().train()
        target_q_net = atari_model(3 * frame_history_len, 11, frame_history_len).cuda().float().eval()   
 
    ''' load old model '''
    optimizer = optim.Adam(train_net.parameters(), lr = args.lr, amsgrad = True)
    dqn_optimizer = optim.Adam(dqn_net.parameters(), lr = args.lr, amsgrad = True)
    replay_buffer = ReplayBuffer(100000, frame_history_len)
     
        
