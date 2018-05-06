from model import *
import torch
import gym
from dqn_utils import *
from mpc_utils import *
import copy
import cv2
from utils import *
from torcs_wrapper import *
from dqn_agent import *

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
    ''' basics '''
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    env = TorcsWrapper(env)

    ''' create model '''
    train_net = ConvLSTMMulti(num_total_act, True, frame_history_len, use_seg, use_xyz, 'drn_d_22', 4, 1024, 32).train()
    net = ConvLSTMMulti(num_total_act, True, frame_history_len, use_seg, use_xyz, 'drn_d_22', 4, 1024, 32).eval()
 
    ''' load old model '''
    optimizer = optim.Adam(train_net.parameters(), lr = args.lr, amsgrad = True)
    mpc_buffer = MPCBuffer(buffer_size, frame_history_len, pred_step, num_total_act, continuous=True, use_xyz = use_xyz, use_seg = use_seg)
    img_buffer = IMGBuffer(1000)
    obs_buffer = ObsBuffer(frame_history_len)
    train_net, epoch, optimizer = load_model(args.save_path, train_net, data_parallel=True, optimizer=optimizer, resume=args.resume)
    net.load_state_dict(train_net.module.state_dict())

    train_net = train_net.cuda()
    net = net.cuda()

    train_net.train()
    net.eval()
    
    ''' environment basics '''
    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
    )

    if use_dqn:
        dqn_agent = DQNAgent(frame_history_len, 11, args.lr, exploration, args.save_path)
        if args.resume:
            dqn_agent.load_model()
        
    epi_rewards, rewards = [], 0.0
    _ = env.reset()
    obs, reward, done, info = env.step(np.array([1.0, 0.0]))
    img_buffer.store_frame(obs)
    prev_info = copy.deepcopy(info)
    avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std(gpu=0)
    speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class=False)
    prev_act = np.array([1.0, 0.0])
    prev_xyz = np.array(info['pos'])

    if args.resume:
        num_imgs_start = max(int(open(args.save_path+'/log_train_torcs.txt').readlines()[-1].split(' ')[1])-1000,0)
    else:
        num_imgs_start = 0

    epi_rewards_with, epi_rewards_without = [], []
    rewards_with, rewards_without = 0, 0
    for tt in range(num_imgs_start, num_steps):
        if use_dqn:
            dqn_action = dqn_agent.sample_action(obs, tt)
        ret = mpc_buffer.store_frame(obs)
        this_obs_np = obs_buffer.store_frame(obs, avg_img, std_img)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0)).float().cuda()

        if random.random() <= 1 - exploration.value(tt):
            ## todo: finish sample continuous action function
            action = sample_cont_action(net, obs_var, prev_action=prev_act, num_time=pred_step)
        else:
            action = np.random.rand(num_total_act) * 2 - 1
        action = np.clip(action, -1, 1)
        action[0] = np.abs(action[0])
        if use_dqn:
            if abs(action[1]) <= dqn_action * 0.1:
                action[1] = 0
        obs, reward, done, info = env.step(action)
        print('action', "{0:.2f}".format(action[0]), "{0:.2f}".format(action[1]), ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),\
            ' reward ', "{0:.2f}".format(reward['with_pos']))
        prev_act = action
        speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)
        offroad_flag, coll_flag = info['off_flag'], info['coll_flag']
        speed_list, pos_list = get_info_ls(prev_info)
        xyz = np.array(info['pos']) if use_xyz else None
        rela_xyz = xyz-prev_xyz
        prev_xyz = xyz
        seg = env.env.get_segmentation().reshape((1,256,256)) if use_seg else None
        mpc_buffer.store_effect(ret, action, done, coll_flag, offroad_flag, speed_list[0], speed_list[1], pos_list[0], rela_xyz, seg)
        rewards_with += reward['with_pos']
        rewards_without += reward['without_pos']

        if tt % 100 == 0:
            avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std()

        if done:
            obs_buffer.clear()
            epi_rewards_with.append(rewards_with)
            epi_rewards_without.append(rewards_without)
            obs = env.reset()
            rewards_with, rewards_without = 0, 0
            obs, reward, done, info = env.step(np.array([1.0, 0.0]))
            prev_act = np.array([1.0, 0.0])
            speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class=False)
            print('past 100 episode rewards is ', \
                "{0:.3f}".format(np.mean(epi_rewards_with[-100:])), \
                ' std is ', "{0:.15f}".format(np.std(epi_rewards_with[-100:])))
            with open(args.save_path+'/log_train_torcs.txt', 'a') as fi:
                fi.write('step ' + str(tt))
                fi.write(' reward_with ' + str(np.mean(epi_rewards_with[-10:])))
                fi.write(' std ' + str(np.std(epi_rewards_with[-10:])))
                fi.write(' reward_without ' + str(np.mean(epi_rewards_without[-10:])))
                fi.write(' std ' + str(np.std(epi_rewards_without[-10:])) + '\n')
        
        prev_info = copy.deepcopy(info) 
        if use_dqn:
            dqn_agent.store_effect(dqn_action, reward['with_pos'], done)
        
        if tt % args.learning_freq == 0 and tt > args.learning_starts and mpc_buffer.can_sample(batch_size):
            for ep in range(10):
                optimizer.zero_grad()
                
                # TODO : FINISH TRAIN MPC MODEL FUNCTION
                loss = train_model(train_net, mpc_buffer, batch_size, epoch, avg_img_t, std_img_t, pred_step, use_xyz, use_seg)
                print('loss = %0.4f' % loss.data.cpu().numpy())
                loss.backward()
                optimizer.step()
                net.load_state_dict(train_net.module.state_dict())
                epoch += 1
                dqn_agent.train_model(batch_size, tt)
                if epoch % save_freq == 0:
                    torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(tt).zfill(9)+'.pt')
                    torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optim_'+str(tt).zfill(9)+'.pt')
                    pkl.dump(epoch, open(args.save_path+'/epoch.pkl', 'wb'))
