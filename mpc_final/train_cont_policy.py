from model import *
import torch
import os
import math
from dqn_utils import *
from mpc_utils import *
import copy
import cv2
from utils import *
from torcs_wrapper import *
from dqn_agent import *
from test import test

def train_policy(args, env, num_steps=40000000):
    ''' basics '''
    env = TorcsWrapper(env, random_reset = args.use_random_reset, continuous = args.continuous)

    if args.target_speed > 0 and os.path.exists(os.path.join(args.save_path, 'speedlog.txt')):
        os.remove(os.path.join(args.save_path, 'speedlog.txt'))
    if args.target_dist > 0 and os.path.exists(os.path.join(args.save_path, 'distlog.txt')):
        os.remove(os.path.join(args.save_path, 'distlog.txt'))

    ''' create model '''
    train_net = ConvLSTMMulti(args)
    net = ConvLSTMMulti(args)
    optimizer = optim.Adam(train_net.parameters(), lr = args.lr, amsgrad = True)

    train_net, epoch, optimizer = load_model(args.save_path, train_net, data_parallel = True, optimizer = optimizer, resume = args.resume)
    net.load_state_dict(train_net.module.state_dict())

    if torch.cuda.is_available():
        train_net = train_net.cuda()
        net = net.cuda()
    train_net.train()
    for param in net.parameters():
        param.requires_grad = False
    net.eval()
 
    ''' load buffers '''
    mpc_buffer = MPCBuffer(args)
    img_buffer = IMGBuffer(1000)
    obs_buffer = ObsBuffer(args.frame_history_len)
    
    ''' environment basics '''
    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
    )

    if args.use_dqn:
        dqn_agent = DQNAgent(args.frame_history_len, args.num_dqn_action, args.lr, exploration, args.save_path, args=args)
        if args.resume:
            dqn_agent.load_model()
    else:
        dqn_agent = None
        
    done_cnt = 0
    epi_rewards, rewards = [], 0.0
    _ = env.reset()
    prev_act = np.array([1.0, 0.0]) if args.continuous else 1
    obs, reward, done, info = env.step(prev_act)
    img_buffer.store_frame(obs)
    prev_info = copy.deepcopy(info)
    avg_img, std_img, avg_img_t, std_img_t = img_buffer.get_avg_std(gpu = 0)
    speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)
    prev_xyz = np.array(info['pos'])

    if args.resume:
        try:
            num_imgs_start = max(int(open(args.save_path + '/log_train_torcs.txt').readlines()[-1].split(' ')[1]) - 1000,0)
        except:
            print('cannot find file, num_imgs_start is 0')
            num_imgs_start = 0
    else:
        num_imgs_start = 0

    epi_rewards_with, epi_rewards_without = [], []
    rewards_with, rewards_without = 0, 0
    start_testing = False
    done_cnt = 0
    for tt in range(num_imgs_start, num_steps):
        if args.use_dqn:
            dqn_action = dqn_agent.sample_action(obs, tt)
        ret = mpc_buffer.store_frame(obs)
        this_obs_np = obs_buffer.store_frame(obs, avg_img, std_img)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0)).float().cuda()

        if tt % args.num_same_step != 0:
            action = prev_act
            real_action = action
            if args.continuous:
                real_action[0] = real_action[0] * 0.5 + 0.5        
        elif args.continuous:
            if random.random() <= 1 - exploration.value(tt):
                ## todo: finish sample continuous action function
                action = sample_cont_action(args, net, obs_var, prev_action = prev_act)
            else:
                action = np.random.rand(args.num_total_act) * 2 - 1
            action = np.clip(action, -1, 1)

            if args.use_dqn:
                if abs(action[1]) <= dqn_action * 0.1:
                    action[1] = 0
            real_action = action
            real_action[0] = real_action[0] * 0.5 + 0.5
        else:
            if random.random() <= 1 - exploration.value(tt):
                real_action = sample_discrete_action(args, net, obs_var, prev_action = prev_act)
            else:
                real_action = np.random.randint(args.num_total_act)
            action = real_action

        obs, reward, done, info = env.step(real_action)
        if args.target_speed > 0:
            with open(os.path.join(args.save_path, 'speedlog.txt'), 'a') as f:
                f.write('step %d speed %0.4f\n' % (tt, info['speed']))
        if args.target_dist > 0:
            with open(os.path.join(args.save_path, 'distlog.txt'), 'a') as f:
                f.write('step %d dist %0.4f\n' % (tt, info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))))
        img_buffer.store_frame(obs)
        if args.continuous:
            print('action', "{0:.2f}".format(action[0]), "{0:.2f}".format(action[1]), ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),\
                ' angle ', "{0:.2f}".format(info['angle']), ' reward ', "{0:.2f}".format(reward['with_pos']), ' explore ', "{0:.2f}".format(exploration.value(tt)))
        else:
            print('action', '%d' % real_action, ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),\
                ' angle ', "{0:.2f}".format(info['angle']), ' reward ', "{0:.2f}".format(reward['with_pos']), ' explore ', "{0:.2f}".format(exploration.value(tt)))
        prev_act = action
        speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)
        offroad_flag, coll_flag = info['off_flag'], info['coll_flag']
        speed_list, pos_list = get_info_ls(prev_info)
        if args.use_xyz:
            xyz = np.array(info['pos'])
            rela_xyz = xyz - prev_xyz
            prev_xyz = xyz
        else:
            rela_xyz = None

        seg = env.env.get_segmentation().reshape((1, 256, 256)) if args.use_seg else None
        mpc_buffer.store_effect(ret, action, done, coll_flag, offroad_flag, info['speed'], info['angle'], pos_list[0], rela_xyz, seg)
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
            prev_act = np.array([1.0, 0.0]) if args.continuous else 1
            obs, reward, done, info = env.step(prev_act)
            speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)
            print('past 100 episode rewards is ', \
                "{0:.3f}".format(np.mean(epi_rewards_with[-100:])), \
                ' std is ', "{0:.15f}".format(np.std(epi_rewards_with[-100:])))
            with open(args.save_path+'/log_train_torcs.txt', 'a') as fi:
                fi.write('step ' + str(tt))
                fi.write(' reward_with ' + str(np.mean(epi_rewards_with[-10:])))
                fi.write(' std ' + str(np.std(epi_rewards_with[-10:])))
                fi.write(' reward_without ' + str(np.mean(epi_rewards_without[-10:])))
                fi.write(' std ' + str(np.std(epi_rewards_without[-10:])) + '\n')
            done_cnt += 1
            if done_cnt % 5 == 0:
                print('begin testing')
                test_reward = test(args, env, net, avg_img, std_img)
                print('Finish testing.')
                with open(os.path.join(args.save_path, 'test_log.txt'), 'a') as f:
                    f.write('step %d reward_with %f reward_without %f\n' % (tt, test_reward['with_pos'], test_reward['without_pos']))
            
        
        prev_info = copy.deepcopy(info) 
        if args.use_dqn:
            dqn_agent.store_effect(dqn_action, reward['with_pos'], done)
        
        if tt % args.learning_freq == 0 and tt > args.learning_starts and mpc_buffer.can_sample(args.batch_size):
            start_testing = True
            for ep in range(50):
                optimizer.zero_grad()
                
                # TODO : FINISH TRAIN MPC MODEL FUNCTION
                loss = train_model(args, train_net, mpc_buffer, epoch, avg_img_t, std_img_t)
                print('loss = %0.4f\n' % loss.data.cpu().numpy())
                loss.backward()
                optimizer.step()
                net.load_state_dict(train_net.module.state_dict())
                epoch += 1

                if args.use_dqn:
                    dqn_agent.train_model(args.batch_size, tt)
                if epoch % args.save_freq == 0:
                    torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(tt).zfill(9)+'.pt')
                    torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optimizer.pt')
                    pkl.dump(epoch, open(args.save_path+'/epoch.pkl', 'wb'))

