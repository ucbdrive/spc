from model import *
import numpy as np
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
import pickle as pkl
import pdb

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
        if args.resume:
            self.mpc_buffer.load(args.save_path)
        self.img_buffer = IMGBuffer()
        self.obs_buffer = ObsBuffer(args.frame_history_len)
        if self.args.lstm2:
            self.action_buffer = ActionBuffer(args.frame_history_len-1)
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
        self.collision_buffer = []
        self.offroad_buffer = []
        self.idx_buffer = []

    def step_first(self, obs, info):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()
        if 'torcs' in self.args.env:
            self.speed_np, self.pos_np, self.posxyz_np = get_info_np(info, use_pos_class=False)
            self.prev_xyz = np.array(info['pos'])


    def store_frame(self, obs, info, seg):
        if self.args.normalize:
            self.img_buffer.store_frame(obs)
        this_obs_np = self.obs_buffer.store_frame(obs)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0).float().cuda())
        self.mpc_ret = self.mpc_buffer.store_frame(obs)
        self.idx_buffer.append(self.mpc_ret)

        if 'torcs' in self.args.env:
            self.speed_np, self.pos_np, self.posxyz_np = get_info_np(info, use_pos_class = False)
            off_flag, coll_flag = info['off_flag'], info['coll_flag']
            _, pos_list = get_info_ls(info)
            if self.args.use_xyz:
                xyz = np.array(info['pos'])
                rela_xyz = xyz - self.prev_xyz
                self.prev_xyz = xyz
            else:
                rela_xyz = None
            self.mpc_buffer.store_effect(idx=self.mpc_ret,
                                         coll=coll_flag,
                                         off=off_flag,
                                         speed=info['speed'],
                                         angle=info['angle'],
                                         pos=pos_list[0],
                                         xyz=rela_xyz,
                                         seg=seg)
        elif 'carla' in self.args.env:
            self.mpc_buffer.store_effect(idx=self.mpc_ret,
                                         coll=info['collision'],
                                         off=info['offroad'],
                                         speed=info['speed'],
                                         otherlane=info['other_lane'],
                                         seg=seg)

        elif 'gta' in self.args.env:
            self.mpc_buffer.store_effect(idx=self.mpc_ret,
                                         coll=info['coll_flag'],
                                         off=info['off_flag'],
                                         speed=info['speed'],
                                         seg=seg)
        return self.mpc_ret, obs_var

    def store_effect(self, guide_action, action, reward, done, collision, offroad):
        self.collision_buffer.append(collision)
        self.offroad_buffer.append(offroad)
        self.prev_act = copy.deepcopy(action)
        act_var = Variable(torch.from_numpy(self.action_buffer.store_frame(action)), requires_grad=False) if self.args.lstm2 else None
        self.mpc_buffer.store_action(self.mpc_ret, guide_action, action, done)
        self.rewards_with += reward['with_pos']
        self.rewards_without += reward['without_pos']
        return act_var

    def update_avg_std_img(self):
        if self.args.normalize:
            self.avg_img, self.std_img = self.img_buffer.get_avg_std()

    def reset(self, info, step, log_name='log_train_torcs.txt'):
        self.obs_buffer.clear()
        self.epi_rewards_with.append(self.rewards_with)
        self.epi_rewards_without.append(self.rewards_without)
        self.rewards_with, self.rewards_without = 0, 0
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1

        if 'torcs' in self.args.env:
            self.speed_np, self.pos_np, self.posxyz_np = get_info_np(info, use_pos_class = False)
            self.prev_xyz = np.array(info['pos'])
        print('past 100 episode rewards is',
              "{0:.3f}".format(np.mean(self.epi_rewards_with[-1:])),
              ' without is ', "{0:.3f}".format(np.mean(self.epi_rewards_without[-1:])))
        with open(self.args.save_path+'/'+log_name, 'a') as fi:
            fi.write('step '+str(step))
            fi.write(' reward_with ' + str(np.mean(self.epi_rewards_with[-1:])))
            fi.write(' std ' + str(np.std(self.epi_rewards_with[-1:])))
            fi.write(' reward_without ' + str(np.mean(self.epi_rewards_without[-1:])))
            fi.write(' std ' + str(np.std(self.epi_rewards_without[-1:])) + '\n')

        epi_len = len(self.idx_buffer)
        idx_buffer = np.array(self.idx_buffer)

        collision_buffer = np.array(self.collision_buffer)
        collision_buffer = np.array([np.sum(collision_buffer[i:i+self.args.safe_length_collision]) == 0 for i in range(collision_buffer.shape[0])])
        offroad_buffer = np.array(self.offroad_buffer)
        offroad_buffer = np.array([np.sum(offroad_buffer[i:i+self.args.safe_length_offroad]) == 0 for i in range(offroad_buffer.shape[0])])
        safe_buffer = collision_buffer * offroad_buffer
        self.mpc_buffer.expert[idx_buffer] = safe_buffer * epi_len
        if np.sum(safe_buffer) > 0:
            self.mpc_buffer.epi_lens.append(epi_len)

        self.idx_buffer = []
        self.collision_buffer = []
        self.offroad_buffer = []

    def save_mpc_buffer(self):
        self.mpc_buffer.save(self.args.save_path)

    def load_mpc_buffer(self):
        self.mpc_buffer.load(self.args.save_path)
        

class ActionSampleManager:
    def __init__(self, args, guides):
        self.args = args
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1
        self.prev_guide_act = 1
        self.guides = guides
        self.p = None

    def sample_action(self, net, dqn_net, obs, obs_var, action_var, exploration, tt, avg_img, std_img, info, no_explore=False, must_explore=False):
        if tt % self.args.num_same_step != 0:
            return self.prev_act, self.prev_guide_act
        else:
            if (random.random() <= 1 - exploration.value(tt) or no_explore) and not must_explore:
                obs = Variable(torch.from_numpy(np.expand_dims(obs.transpose(2, 0, 1), axis=0)).float())
                if torch.cuda.is_available():
                    obs = obs.cuda()
                with torch.no_grad():
                    obs = obs.repeat(max(1, torch.cuda.device_count()), 1, 1, 1)
                    p = F.softmax(net(obs, function='guide_action'), dim=1)[0].data.cpu().numpy()
                    self.p = p
                action = sample_cont_action(self.args, p, net, obs_var, self.guides, info=info, prev_action=np.array([0.5, 0.01]), avg_img=avg_img, std_img=std_img, tt=tt, action_var=action_var)
            else:
                self.p = None
                action = np.random.uniform(-1, 1, size=(self.args.num_total_act,))  # generate_action(p, size=1).reshape(-1)
            action = np.clip(action, -1, 1)
            guide_act = get_guide_action(self.args.bin_divide, action)
            self.prev_act = action
            self.prev_guide_act = guide_act
            return action, guide_act

    def reset(self):
        self.prev_act = np.array([1.0, -0.1]) if self.args.continuous else 1
        self.prev_guide_act = 1

    def process_act(self, act, dqn_act):
        if self.args.use_dqn and self.args.continuous:
            if abs(act[1]) <= dqn_act * 0.1:
                act[1] = 0
        elif self.args.continuous and not self.args.use_dqn:
            if abs(act[1]) <= 0.0:
                act[1] = 0
        return act, dqn_act


def train_policy(args, env, num_steps=40000000):
    guides = generate_guide_grid(args.bin_divide)

    ''' basics '''
    if 'torcs' in args.env:
        env = TorcsWrapper(env, random_reset=args.use_random_reset, continuous = args.continuous)

    ''' create model '''
    train_net, net, optimizer, epoch, exploration, dqn_agent, num_imgs_start = init_models(args)

    ''' load buffers '''
    buffer_manager = BufferManager(args)
    action_manager = ActionSampleManager(args, guides)

    video_folder = os.path.join(args.video_folder, "%d" % num_imgs_start)
    if not os.path.isdir(video_folder):
        os.makedirs(video_folder)
    video = cv2.VideoWriter(os.path.join(video_folder, 'video.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (256, 256), True)

    done_cnt = 0
    (obs, seg), info = env.reset()
    video.write(obs)
    obs, reward, done, info = env.step(buffer_manager.prev_act)
    if 'carla' in args.env:
        obs, seg = obs
    buffer_manager.step_first(obs, info)
    video.write(obs)
    with open(os.path.join(video_folder, 'actions.txt'), 'a') as f:
        _guide_act = guides[get_guide_action(args.bin_divide, buffer_manager.prev_act)]
        f.write('affordance %0.2f %0.2f action %0.4f %0.4f\n' % (
            _guide_act[0],
            _guide_act[1],
            buffer_manager.prev_act[0],
            buffer_manager.prev_act[1]
        ))

    done_cnt = 0
    no_explore = False
    num_episode = 0
    must_explore = False
    print('start game')
    action_var = Variable(torch.from_numpy(np.array([-1.0, 0.0])).repeat(1, args.frame_history_len - 1, 1), requires_grad=False)
    for tt in range(num_imgs_start, num_steps):
        if 'torcs' in args.env:
            seg = env.env.get_segmentation().reshape((1, 256, 256)) if args.use_seg else None
        if 'gta' in args.env:
            seg = env.get_segmentation().reshape((1, 256, 256)) if args.use_seg else None

        ret, obs_var = buffer_manager.store_frame(obs, info, seg)
        if args.normalize:
            avg_img, std_img = buffer_manager.img_buffer.get_avg_std()
        else:
            avg_img, std_img = None, None

        action, guide_action = action_manager.sample_action(net, dqn_agent, obs, obs_var, action_var, exploration, tt, avg_img, std_img, info, no_explore=no_explore, must_explore=must_explore)

        obs, reward, done, info = env.step(action)
        if 'carla' in args.env:
            obs, seg = obs
        if args.target_speed > 0:
            with open(os.path.join(args.save_path, 'speedlog.txt'), 'a') as f:
                f.write('step %d speed %0.4f target %0.4f\n' % (tt, info['speed'], args.target_speed))
        if args.target_dist > 0:
            with open(os.path.join(args.save_path, 'distlog.txt'), 'a') as f:
                f.write('step %d dist %0.4f\n' % (tt, info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))))
        if 'torcs' in args.env:
            if args.continuous:
                print('action', "{0:.2f}".format(action[0]), "{0:.2f}".format(action[1]), ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),
                      ' angle ', "{0:.2f}".format(info['angle']), ' reward ', "{0:.2f}".format(reward['with_pos']), ' explore ', "{0:.2f}".format(exploration.value(tt)))
            else:
                print('action', '%d' % action, ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),
                      ' angle ', "{0:.2f}".format(info['angle']), ' reward ', "{0:.2f}".format(reward['with_pos']), ' explore ', "{0:.2f}".format(exploration.value(tt)))
        elif 'carla' in args.env:
            print('action', "{0:.2f}".format(action[0]), "{0:.2f}".format(action[1]),
                  ' collision ', str(bool(info['collision'])),
                  ' offroad ', str(bool(info['offroad'])),  # "{0:.2f}%".format(info['offroad']*100.0),
                  ' otherlane ', "{0:.2f}%".format(info['other_lane']*100.0),
                  ' speed ', "{0:.2f}".format(info['speed']),
                  ' reward_with_pos ', "{0:.2f}".format(reward['with_pos']),
                  ' reward_without_pos ', "{0:.2f}".format(reward['without_pos']),
                  ' explore ', "{0:.2f}".format(exploration.value(tt)))
        elif 'gta' in args.env:
            print('action', "{0:.2f}".format(action[0]), "{0:.2f}".format(action[1]),
                  ' collision ', str(bool(info['coll_flag'])),
                  ' offroad ', str(bool(info['off_flag'])),  # "{0:.2f}%".format(info['offroad']*100.0),
                  ' speed ', "{0:.2f}".format(info['speed']),
                  ' reward_with_pos ', "{0:.2f}".format(reward['with_pos']),
                  ' reward_without_pos ', "{0:.2f}".format(reward['without_pos']),
                  ' explore ', "{0:.2f}".format(exploration.value(tt)))

        action_var = buffer_manager.store_effect(guide_action, action, reward, done, info['collision'], info['offroad'])
        # if action_manager.p is not None:
        #     obs = draw_action2(args, obs, 150, 66, 45, action_manager.p.reshape(-1))
        with open(os.path.join(video_folder, 'actions.txt'), 'a') as f:
            _guide_act = guides[get_guide_action(args.bin_divide, action)]
            f.write('affordance %0.2f %0.2f action %0.4f %0.4f\n' % (
                _guide_act[0],
                _guide_act[1],
                action[0],
                action[1]
            ))

        if tt % 100 == 0 and args.normalize:
            buffer_manager.update_avg_std_img()

        if tt % args.learning_freq == 0 and buffer_manager.mpc_buffer.can_sample(args.batch_size):
            for ep in range(args.num_train_steps):
                optimizer.zero_grad()
                loss = train_model(args, train_net, buffer_manager.mpc_buffer, epoch, buffer_manager.avg_img, buffer_manager.std_img) + train_guide_action(args, train_net, buffer_manager.mpc_buffer, guides)
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
                buffer_manager.save_mpc_buffer()
                print('\033[1;32mModels saved successfully!\033[0m')


        if done:
            num_episode += 1
            print('finished episode ', num_episode)

            video.release()
            # os.system('ffmpeg -y -i %s %s' % (os.path.join(video_folder, 'video.avi'), os.path.join(video_folder, 'video.mp4')))
            # os.remove(os.path.join(video_folder, 'video.avi'))
            video_folder = os.path.join(args.video_folder, "%d" % tt)
            if not os.path.isdir(video_folder):
                os.makedirs(video_folder)
            video = cv2.VideoWriter(os.path.join(video_folder, 'video.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (256, 256), True)

            no_explore = not no_explore
            done_cnt += 1
            if 'torcs' in args.env:
                obs, prev_info = env.reset(restart=True)
            elif 'carla' in args.env:
                obs, prev_info = env.reset(testing=no_explore)
            elif 'gta' in args.env:
                obs, prev_info = env.reset()
                # print('reset!!! ')
            obs, _, _, info = env.step(np.array([-1.0, 0.0])) if args.continuous else env.step(1)
            if 'carla' in args.env:
                obs, seg = obs
            buffer_manager.reset(prev_info, tt)
            action_manager.reset()
            if args.target_speed > 0:
                args.target_speed = np.random.uniform(20, 30)