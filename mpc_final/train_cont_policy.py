from model import *
import torch
import gym
from dqn_utils import *
from mpc_utils import *
import copy
import cv2
from utils import *

class DQNAgent:
    def __init__(self, frame_history_len=4, num_actions=11, lr=0.0001, exploration=None, save_path=None):
        self.dqn_net = atari_model(3 * frame_history_len, num_actions, frame_history_len).cuda().float().train()
        self.target_q_net = atari_model(3 * frame_history_len, num_actions, frame_history_len).cuda().float().eval()
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr = lr, amsgrad=True)
        self.replay_buffer = ReplayBuffer(100000, frame_history_len)
        self.frame_history_len = frame_history_len
        self.num_actions = num_actions
        self.ret = 0
        self.exploration = exploration
        self.num_param_updates = 0
        self.save_path = save_path
        if os.path.isdir(os.path.join(save_path, 'dqn')) == False:
            os.mkdir(os.path.join(save_path, 'dqn'))
        if os.path.isdir(os.path.join(save_path, 'dqn', 'model')) == False:
            os.mkdir(os.path.join(save_path, 'dqn', 'model'))
        if os.path.isdir(os.path.join(save_path, 'dqn', 'optimizer')) == False:
            os.mkdir(os.path.join(save_path, 'dqn', 'optimizer'))
        self.model_path = os.path.join(save_path, 'dqn', 'model')
        self.optim_path = os.path.join(save_path, 'dqn', 'optimizer')

    def load_model(self):
        model_path, optim_path = self.model_path, self.optim_path
        file_list = sorted(os.listdir(model_path))
        file_name = os.path.join(model_path, file_list[-1])
        self.dqn_net.load_state_dict(torch.load(os.path.join(file_name)))
        self.target_q_net.load_state_dict(torch.load(os.path.join(file_name)))
        
        optim_list = sorted(os.listdir(optim_path))
        optim_name = os.path.join(optim_path, optim_list[-1])
        self.optimizer.load_state_dict(torch.load(os.path.join(optim_name)))
        
    def sample_action(self, obs, t):
        self.ret = self.replay_buffer.store_frame(cv2.resize(obs, (84, 84)))
        dqn_obs = self.replay_buffer.encode_recent_observation()
        sample = random.random()
        eps_threshold = self.exploration.value(t)
        if sample > eps_threshold:
            with torch.no_grad():
                dqn_obs = torch.from_numpy(dqn_obs).cuda().float().unsqueeze(0) / 255.0
                action = int(self.dqn_net(Variable(dqn_obs)).data.max(1)[1].cpu().numpy())
        else:
            action = random.randrange(self.num_actions)
        return int(action)

    def store_effect(self, action, reward, done):
        self.replay_buffer.store_effect(self.ret, action, reward, done)

    def train_model(self, batchsize, save_num=None):
        dtype = torch.cuda.FloatTensor
        obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(batch_size)
        obs_t_batch     = Variable(torch.from_numpy(obs_t_batch).type(dtype) / 255.0)
        act_batch       = Variable(torch.from_numpy(act_batch).long()).cuda()
        rew_batch       = Variable(torch.from_numpy(rew_batch)).cuda()
        obs_tp1_batch   = Variable(torch.from_numpy(obs_tp1_batch).type(dtype) / 255.0)
        done_mask       = Variable(torch.from_numpy(done_mask)).type(dtype)
       
        q_a_values = self.dqn_net(obs_t_batch).gather(1, act_batch.unsqueeze(1))
        q_a_values_tp1 = self.target_q_net(obs_tp1_batch).detach().max(1)[0]
        target_values = rew_batch + (0.99 * (1-done_mask) * q_a_values_tp1)
        dqn_loss = ((target_values.view(q_a_values.size()) - q_a_values)**2).mean()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        self.num_param_updates += 1
        
        if self.num_param_updates % 100 == 0:
            self.target_q_net.load_state_dict(self.dqn_net.state_dict())
        torch.save(self.target_q_net.state_dict(), self.model_path+'/model_'+str(save_num)+'.pt')
        torch.save(self.optimizer.state_dict(), self.optim_path+'/optim_'+str(save_num)+'.pt')
 
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
    mpc_buffer = MPCBuffer(buffer_size, frame_history_len, pred_step, num_total_act, continuous=True)
    img_buffer = IMGBuffer(1000)
    obs_buffer = ObsBuffer(frame_history_len)
    train_net, epoch, optimizer = load_model(args.save_path, train_net, data_parallel=True, optimizer=optimizer, resume=args.resume)
    net.load_state_dict(train_net.module.state_dict())
    train_net.cuda().float().train()
    net.float().cuda().eval()
    
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
        rand_num = random.random()
        if rand_num <= 1-exploration.value(tt):
            ## todo: finish sample continuous action function
            action = sample_cont_action(net, obs_var, prev_action=prev_act, num_time=pred_step)
        else:
            action = np.random.rand(2)*2-1
        action = np.clip(action, -1, 1)
        if use_dqn:
            if abs(action[1]) <= dqn_action * 0.1:
                action[1] = 0
        obs, reward, done, info = env.step(action)
        prev_act = action
        speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)
        offroad_flag, coll_flag = info['off_flag'], info['coll_flag']
        speed_list, pos_list = get_info_ls(prev_info)
        mpc_buffer.store_effect(ret, action, done, coll_flag, offroad_flag, speed_list[0], speed_list[1], pos_list[0])
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
                fi.write('step '+str(tt)+' reward_with '+str(np.mean(epi_rewards_with[-10:]))+' std '+str(np.std(epi_rewards_with[-10:]))+\
                        ' reward_without '+str(np.mean(epi_rewards_without[-10:]))+' std '+str(np.std(epi_rewards_without)[-10:]))+'\n')
        
        prev_info = copy.deepcopy(info) 
        if use_dqn:
            dqn_agent.store_effect(dqn_action, reward['with_pos'], done)
        
        if tt % args.learning_freq == 0 and tt > args.learning_starts and mpc_buffer.can_sample(batch_size):
            for ep in range(10):
                optimizer.zero_grad()
                
                # TODO : FINISH TRAIN MPC MODEL FUNCTION
                loss = train_model(train_net, mpc_buffer, batch_size, epoch, avg_img_t, std_img_t, pred_step)
                loss.backward()
                optimizer.step()
                net.load_state_dict(train_net.module.state_dict())
                epoch += 1
                if epoch % save_freq == 0:
                    torch.save(train_net.module.state_dict(), args.save_path+'/model/pred_model_'+str(tt).zfill(9)+'.pt')
                    torch.save(optimizer.state_dict(), args.save_path+'/optimizer/optim_'+str(tt).zfill(9)+'.pt')
                    pkl.dump(epoch, open(args.save_path+'/epoch.pkl', 'wb'))
                    dqn_agent.train_model(batch_size, tt) 
