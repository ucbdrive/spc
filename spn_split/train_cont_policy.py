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
import pdb

def init_models(args):
    train_net = ConvLSTMMulti(args)
    for param in train_net.parameters():
        param.requires_grad = True
    train_net.train()
    net = ConvLSTMMulti(args)
    optimizer = optim.Adam(train_net.parameters(), lr=args.lr, amsgrad=True)
    train_net, epoch, optimizer = load_model(args.save_path, train_net, data_parallel=args.data_parallel, optimizer=optimizer, resume=args.resume)
    if args.data_parallel:
        net.load_state_dict(train_net.module.state_dict())
    else:
        net.load_state_dict(train_net.state_dict())
    
    if torch.cuda.is_available():
        train_net = train_net.cuda()
        net = net.cuda()
    for param in train_net.parameters():
        param.requires_grad = True
    train_net.train()
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value = 0.02
    )

    if args.use_dqn:
        dqn_agent = DQNAgent(args, exploration, args.save_path)
        if args.resume:
            dqn_agent.load_model()
    else:
        dqn_agent = None 

    if args.resume:
        try:
            num_imgs_start = max(int(open(args.save_path + '/log_train_torcs.txt').readlines()[-1].split(' ')[1]) - 1000,0)
        except:
            print('cannot find file, num_imgs_start is 0')
            num_imgs_start = 0
    else:
        num_imgs_start = 0

    return train_net, net, optimizer, epoch, exploration, dqn_agent, num_imgs_start

class BufferManager:
    def __init__(self, args=None):
        self.args = args
        self.mpc_buffer = MPCBuffer(args)
        self.img_buffer = IMGBuffer()
        self.obs_buffer = ObsBuffer(args.frame_history_len)
        self.epi_rewards = []
        self.rewards = 0.0
        self.prev_act = np.array([1.0, -0.1]) if args.continuous else 1
        
        self.avg_img = None
        self.std_img = None
        self.speed_np = None
        self.pos_np = None
        self.posxyz_np = None
        self.prev_xyz = None
        self.epi_rewards_with = []
        self.epi_rewards_without = []
        self.rewards_with = 0.0
        self.rewards_without = 0.0
        self.mpc_ret = 0
        
    def step_first(self, obs, info):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()
        self.speed_np, self.pos_np, self.posxyz_np = get_info_np(info, use_pos_class=False)
        self.prev_xyz = np.array(info['pos'])

    def store_frame(self, obs, info, seg):
        self.mpc_ret = self.mpc_buffer.store_frame(obs)
        self.speed_np, self.pos_np, self.posxyz_np = get_info_np(info, use_pos_class = False)
        off_flag, coll_flag = info['off_flag'], info['coll_flag']
        speed_list, pos_list = get_info_ls(info)
        if self.args.use_xyz:
            xyz = np.array(info['pos'])
            rela_xyz = xyz - self.prev_xyz
            self.prev_xyz = xyz
        else:
            rela_xyz = None
        self.mpc_buffer.store_effect(self.mpc_ret, coll_flag, off_flag, info['speed'], info['angle'], pos_list[0], rela_xyz, seg)
        this_obs_np = self.obs_buffer.store_frame(obs)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0).float().cuda())
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
        return self.mpc_ret, obs_var
    
    def store_effect(self, action, reward, done):
        self.prev_act = copy.deepcopy(action)
        self.mpc_buffer.store_action(self.mpc_ret, action, done)
        self.rewards_with += reward['with_pos']
        self.rewards_without += reward['without_pos']

    def update_avg_std_img(self):
        if self.args.normalize:
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()
    
    def reset(self, info, step, log_name='log_train_torcs.txt'):
        self.obs_buffer.clear()
        self.epi_rewards_with.append(self.rewards_with)
        self.epi_rewards_without.append(self.rewards_without)
        self.rewards_with, self.rewards_without = 0, 0
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1
        self.speed_np, self.pos_np, self.posxyz_np = get_info_np(info, use_pos_class = False)
        self.prev_xyz = np.array(info['pos'])
        print('past 100 episode rewards is', \
            "{0:.3f}".format(np.mean(self.epi_rewards_with[-1:])), \
                ' without is ', "{0:.3f}".format(np.mean(self.epi_rewards_without[-1:])))
        with open(self.args.save_path+'/'+log_name, 'a') as fi:
            fi.write('step '+str(step))
            fi.write(' reward_with ' + str(np.mean(self.epi_rewards_with[-1:])))
            fi.write(' std ' + str(np.std(self.epi_rewards_with[-1:])))
            fi.write(' reward_without ' + str(np.mean(self.epi_rewards_without[-1:])))
            fi.write(' std ' + str(np.std(self.epi_rewards_without[-1:])) + '\n')      

class ActionSampleManager:
    def __init__(self, args):
        self.args = args
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def sample_action(self, net, dqn_net, obs, obs_var, exploration, tt, avg_img, std_img, info, no_explore=False, must_explore=False):
        if tt % self.args.num_same_step != 0:
            return self.process_act(self.prev_act, self.prev_dqn_act)
        else:
            if self.args.continuous:
                if (random.random() <= 1 - exploration.value(tt) or no_explore) and must_explore == False:
                    action = sample_cont_action(self.args, net, obs_var, info=info, prev_action=np.array([0.5, 0.01]), avg_img=avg_img, std_img=std_img, tt=tt)
                else:
                    action = np.random.rand(self.args.num_total_act)*2-1
                action = np.clip(action, -1, 1)
                if self.args.use_dqn:
                    dqn_act = dqn_net.sample_action(obs, tt)
                else:
                    dqn_act = None
            else:
                if random.random() <= 1- exploration.value(tt):
                    with torch.no_grad():
                        action = sample_discrete_action(self.args, net, obs_var, prev_action=self.prev_act)
                else:
                    action = np.random.randint(self.args.num_total_act)
                dqn_act = None
            action, dqn_act = self.process_act(action, dqn_act)
            self.prev_act = action
            self.prev_dqn_act = dqn_act
            return action, dqn_act               

    def reset(self):
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None
 
    def process_act(self, act, dqn_act):
        if self.args.use_dqn and self.args.continuous:
            if abs(act[1]) <= dqn_act * 0.1:
                act[1] = 0
        elif self.args.continuous and self.args.use_dqn == False:
            if abs(act[1]) <= 0.0:
                act[1] = 0
        return act, dqn_act        

def train_policy(args, env, num_steps=40000000):
    ''' basics '''
    env = TorcsWrapper(env, random_reset = args.use_random_reset, continuous = args.continuous)

    ''' create model '''
    train_net, net, optimizer, epoch, exploration, dqn_agent, num_imgs_start = init_models(args)
    
    ''' load buffers '''
    buffer_manager = BufferManager(args)
    action_manager = ActionSampleManager(args)

    done_cnt = 0
    _, info = env.reset()
    obs, reward, done, info = env.step(buffer_manager.prev_act)
    buffer_manager.step_first(obs, info)
    done_cnt = 0
    no_explore = False
    num_episode = 0
    must_explore = False
    for tt in range(num_imgs_start, num_steps):
        seg = env.env.get_segmentation().reshape((1, 256, 256)) if args.use_seg else None
        ret, obs_var = buffer_manager.store_frame(obs, info, seg)
        if args.normalize:
            avg_img, std_img = buffer_manager.img_buffer.get_avg_std()
        else:
            avg_img, std_img = None, None
        # if info['trackPos'] < -7.0:
        #     must_explore = True
        # else:
        #     must_explore = False
        action, dqn_action = action_manager.sample_action(net, dqn_agent, obs, obs_var, exploration, tt, avg_img, std_img, info, no_explore=no_explore, must_explore=must_explore)
        #if num_episode % 3 == 0:
        #    action[1] = action[1]*-1.0
        obs, reward, done, info = env.step(action)
        if args.target_speed > 0:
            with open(os.path.join(args.save_path, 'speedlog.txt'), 'a') as f:
                f.write('step %d speed %0.4f target %0.4f\n' % (tt, info['speed'], args.target_speed))
        if args.target_dist > 0:
            with open(os.path.join(args.save_path, 'distlog.txt'), 'a') as f:
                f.write('step %d dist %0.4f\n' % (tt, info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))))
        if args.continuous:
            print('action', "{0:.2f}".format(action[0]), "{0:.2f}".format(action[1]), ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),\
                ' angle ', "{0:.2f}".format(info['angle']), ' reward ', "{0:.2f}".format(reward['with_pos']), ' explore ', "{0:.2f}".format(exploration.value(tt)))
        else:
            print('action', '%d' % action, ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),\
                ' angle ', "{0:.2f}".format(info['angle']), ' reward ', "{0:.2f}".format(reward['with_pos']), ' explore ', "{0:.2f}".format(exploration.value(tt)))
        
        buffer_manager.store_effect(action, reward, done)

        if tt % 100 == 0 and args.normalize:
            buffer_manager.update_avg_std_img()

        if done:
            train_model_new(args, train_net, buffer_manager.mpc_buffer, optimizer, tt)
            num_episode += 1
            print('finished episode ', num_episode)
            if no_explore:
                no_explore = False
            else:
                no_explore = True
            done_cnt += 1
            if num_episode % 5 == 0:
                obs, prev_info = env.reset(restart=True)
            else:
                obs, prev_info = env.reset(restart=True)
            obs, _, _, info = env.step(np.array([1.0, -0.1])) if args.continuous else env.step(1)
            buffer_manager.reset(prev_info, tt)
            action_manager.reset()
            if args.target_speed >0:
                args.target_speed = np.random.uniform(20, 30)
        
        if args.use_dqn:
            dqn_agent.store_effect(dqn_action, reward['with_pos'], done)
        
        if tt % args.learning_freq == 0 and tt > args.learning_starts and buffer_manager.mpc_buffer.can_sample(args.batch_size):
            # train_model_new(args, train_net, buffer_manager.mpc_buffer, optimizer, tt)
            # for ep in range(args.num_train_steps):
                # optimizer.zero_grad()
                # loss = train_model(args, train_net, buffer_manager.mpc_buffer, epoch, buffer_manager.avg_img, buffer_manager.std_img)
                # print('loss = %0.4f\n' % loss.data.cpu().numpy())
                # loss.backward()
                # optimizer.step()
                # epoch += 1
            if args.data_parallel:
                net.load_state_dict(train_net.module.state_dict())
            else:
                net.load_state_dict(train_net.state_dict())

            if args.use_dqn:
                dqn_agent.train_model(args.batch_size, tt)
            if True:#epoch % args.save_freq == 0:
                torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(tt).zfill(9)+'.pt')
                torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optimizer.pt')
                pkl.dump(epoch, open(args.save_path+'/epoch.pkl', 'wb'))
