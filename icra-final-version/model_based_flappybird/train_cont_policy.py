from model import ConvLSTMMulti
import numpy as np
import random
import os
import torch
from torch.autograd import Variable
from dqn_utils import PiecewiseSchedule
from mpc_utils import MPCBuffer, IMGBuffer
import copy
from torch import optim
from utils import load_model, ObsBuffer, ActionBuffer, sample_cont_action, train_model, sample_discrete_action
from dqn_agent import DQNAgent
import pickle as pkl


def init_models(args):
    train_net = ConvLSTMMulti(args)
    for param in train_net.parameters():
        param.requires_grad = True
    train_net.train()

    net = ConvLSTMMulti(args)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    train_net, epoch = load_model(args.save_path, train_net, data_parallel=args.data_parallel, resume=args.resume)
    net.load_state_dict(train_net.state_dict())

    if torch.cuda.is_available():
        train_net = train_net.cuda()
        net = net.cuda()
        if args.data_parallel:
            train_net = torch.nn.DataParallel(train_net)
            net = torch.nn.DataParallel(net)
    optimizer = optim.Adam(train_net.parameters(), lr=args.lr, amsgrad=True)

    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
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
        # if args.resume:
        #     self.mpc_buffer.load(args.save_path)
        self.img_buffer = IMGBuffer()
        self.obs_buffer = ObsBuffer(args.frame_history_len)
        if self.args.lstm2:
            self.action_buffer = ActionBuffer(args.frame_history_len-1)
        self.prev_act = np.array([0.0]) if args.continuous else 1

        self.avg_img = None
        self.std_img = None
        self.epi_rewards = []
        self.rewards = 0.0
        self.mpc_ret = 0

        self._rewards = []
        self.reward_idx = []

    def step_first(self, obs, info):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()

    def store_frame(self, obs, reward, seg):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
        this_obs_np = self.obs_buffer.store_frame(obs)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0).float().cuda())
        self._rewards.append(reward)
        self.reward_idx.append(self.mpc_ret)
        self.mpc_ret = self.mpc_buffer.store_frame(obs)
        self.rewards += reward

        self.mpc_buffer.store_effect(idx=self.mpc_ret,
                                     reward=reward,
                                     seg=seg)
        return self.mpc_ret, obs_var

    def store_effect(self, action, reward, done):
        self.prev_act = copy.deepcopy(action)
        act_var = Variable(torch.from_numpy(self.action_buffer.store_frame(action)), requires_grad=False) if self.args.lstm2 else None
        self.mpc_buffer.store_action(self.mpc_ret, action, done)
        if done:
            for i in range(len(self._rewards)-1, 0, -1):
                self._rewards[i-1] += self.args.gamma * self._rewards[i]
            for i in range(len(self._rewards)):
                self.mpc_buffer.value[self.reward_idx[i], 0] = self._rewards[i]
            self._rewards = []
            self.reward_idx = []
        return act_var

    def update_avg_std_img(self):
        if self.args.normalize:
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()

    def reset(self, step, log_name='log_train_torcs.txt'):
        self.obs_buffer.clear()
        self.epi_rewards.append(self.rewards)
        self.rewards = 0
        self.prev_act = np.array([0.0]) if self.args.continuous else 1
        print('past 100 episode rewards is',
              "{0:.3f}".format(np.mean(self.epi_rewards[-1:])))
        with open(os.path.join(self.args.save_path, log_name), 'a') as fi:
            fi.write('step '+str(step))
            fi.write(' reward ' + str(np.mean(self.epi_rewards[-1:])))
            fi.write(' std ' + str(np.std(self.epi_rewards[-1:])))
            fi.write('\n')

    def save_mpc_buffer(self):
        self.mpc_buffer.save(self.args.save_path)

    def load_mpc_buffer(self):
        self.mpc_buffer.load(self.args.save_path)


class ActionSampleManager:
    def __init__(self, args):
        self.args = args
        self.prev_act = np.array([0.0]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def sample_action(self, net, dqn_net, obs, obs_var, action_var, exploration, tt, avg_img, std_img, info, no_explore=False, must_explore=False):
        if tt % self.args.num_same_step != 0:
            return self.process_act(self.prev_act, self.prev_dqn_act)
        else:
            if self.args.continuous:
                if (random.random() <= 1 - exploration.value(tt) or no_explore) and not must_explore:
                    action = sample_cont_action(self.args, net, obs_var, info=info, prev_action=np.array([0.0]), avg_img=avg_img, std_img=std_img, tt=tt, action_var=action_var)
                else:
                    action = np.random.rand(self.args.num_total_act)*2-1
                action = np.clip(action, -1, 1)
                if self.args.use_dqn:
                    dqn_act = dqn_net.sample_action(obs, tt)
                else:
                    dqn_act = None
            else:
                if random.random() <= 1 - exploration.value(tt):
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
        self.prev_act = np.array([0.0]) if self.args.continuous else 1
        if self.args.use_dqn:
            self.prev_dqn_act = 0
        else:
            self.prev_dqn_act = None

    def process_act(self, act, dqn_act):
        return act, dqn_act


def train_policy(args, env, num_steps=40000000):
    ''' create model '''
    train_net, net, optimizer, epoch, exploration, dqn_agent, num_imgs_start = init_models(args)

    ''' load buffers '''
    buffer_manager = BufferManager(args)
    action_manager = ActionSampleManager(args)

    done_cnt = 0
    _, info = env.reset()
    obs, reward, done, info = env.step(buffer_manager.prev_act)
    seg = info['segmentation']
    buffer_manager.step_first(obs, info)
    done_cnt = 0
    no_explore = False
    num_episode = 0
    must_explore = False
    print('start game')
    action_var = Variable(torch.from_numpy(np.array([0.0])).repeat(1, args.frame_history_len - 1, 1), requires_grad=False)
    for tt in range(num_imgs_start, num_steps):
        ret, obs_var = buffer_manager.store_frame(obs, reward, seg)
        if args.normalize:
            avg_img, std_img = buffer_manager.img_buffer.get_avg_std()
        else:
            avg_img, std_img = None, None
        action, dqn_action = action_manager.sample_action(net, dqn_agent, obs, obs_var, action_var, exploration, tt, avg_img, std_img, info, no_explore=no_explore, must_explore=must_explore)

        obs, reward, done, info = env.step(action)
        seg = info['segmentation']
        print('Action = %d, Reward = %d, Done = %s' % (int(action), reward, str(done)))
        action_var = buffer_manager.store_effect(action, reward, done)

        if tt % 100 == 0 and args.normalize:
            buffer_manager.update_avg_std_img()

        if done:
            print('done, episode terminates')

        if tt % args.learning_freq == 0 and tt > args.learning_starts and buffer_manager.mpc_buffer.can_sample(args.batch_size):
            # train_model_new(args, train_net, buffer_manager.mpc_buffer, tt)
            for ep in range(args.num_train_steps):
                optimizer.zero_grad()
                loss = train_model(args, train_net, buffer_manager.mpc_buffer, epoch, buffer_manager.avg_img, buffer_manager.std_img)
                print('loss = %0.4f\n' % loss.data.cpu().numpy())
                loss.backward()
                optimizer.step()
                epoch += 1
            net.load_state_dict(train_net.state_dict())

            if args.use_dqn:
                dqn_agent.train_model(args.batch_size, tt)
            if epoch % args.save_freq == 0:
                print('\033[1;32mSaving models, please wait......\033[0m')
                torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(tt).zfill(9)+'.pt')
                torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optimizer.pt')
                pkl.dump(epoch, open(args.save_path+'/epoch.pkl', 'wb'))
                # buffer_manager.save_mpc_buffer()
                print('\033[1;32mModels saved successfully!\033[0m')

        if done:
            # train_model_new(args, train_net, buffer_manager.mpc_buffer, optimizer, tt)
            num_episode += 1
            print('finished episode ', num_episode)
            no_explore = not no_explore
            done_cnt += 1
            obs, info = env.reset()
            obs, _, _, info = env.step(np.array([0.0])) if args.continuous else env.step(1)
            seg = info['segmentation']
            buffer_manager.reset(tt)
            action_manager.reset()
            if args.target_speed > 0:
                args.target_speed = np.random.uniform(20, 30)

        if args.use_dqn:
            dqn_agent.store_effect(dqn_action, reward['with_pos'], done)
