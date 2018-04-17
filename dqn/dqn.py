import sys
import os
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from dqn_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import PIL.Image as Image
import pdb
import cv2
import copy

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

def learn(args,
          env,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          use_cuda=True,
          global_lock=None,
          global_step=None,
          load_old_q_value=True,
          lr_schedule=None):
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    if use_cuda == True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    if len(env.observation_space.shape) == 1:
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = args.num_total_act

    q_value = q_func(input_arg, num_actions, without_dla=args.without_dla).type(dtype)
    target_q_value = q_func(input_arg, num_actions, without_dla=args.without_dla).type(dtype)
    if load_old_q_value == True:
        model_paths = os.listdir(args.save_path+'/models')
        if len(model_paths) <= 1:
            pass 
        else:
            model_paths_new = [int(model_paths[i]) for i in range(len(model_paths))]
            model_paths_new = sorted(model_paths_new)
            load_path = os.path.join(args.save_path, 'models', str(model_paths_new[-1]))
            print('load model ', str(model_paths_new[-1]))
            with global_lock:
                global_step.value = int(model_paths_new[-1])
            q_value.load_state_dict(torch.load(load_path+'/model_0.pt'))
            target_q_value.load_state_dict(torch.load(load_path+'/model_0.pt'))
    
    optimizer = optimizer_spec.constructor(q_value.parameters(), **optimizer_spec.kwargs)
    q_value.eval()
    target_q_value.eval()
 
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            action = int(model(Variable(obs, requires_grad=False)).data.max(1)[1].cpu().numpy())
        else:
            action = random.randrange(num_actions)
        return torch.IntTensor([[action]])

    def select_attack_action(model, t_model, obs, t, n=20, alpha=0.3, beta=0.8):
        ''' select attacked action '''
        sample = random.random()
        eps_th = exploration.value(t)
        if sample > eps_th:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255.0
            a_star = int(model(Variable(obs, volatile=True)).data.max(1)[1].cpu().numpy())
            Q_star = float(t_model(Variable(obs, requires_grad=False)).data.max(1)[0].cpu().numpy())
            obs_var = Variable(obs, requires_grad=True)
            pi_target = torch.nn.Softmax()(t_model(obs_var))
            worst_action = int(pi_target.data.min(1)[1].cpu().numpy())
            j_loss = -1.0*torch.log(pi_target[0,worst_action])
            j_loss.backward()
            grad = obs_var.grad
            grad_dir = grad/torch.norm(grad, 2)
            s_adv = obs_var
            for i in range(n):
                ni = np.random.beta(alpha,beta)
                si = obs_var - ni*grad_dir
                a_adv = int(model(si).data.max(1)[1].cpu().numpy())
                q_target_adv = float(t_model(obs_var).data.cpu().numpy()[0,a_adv])
                if q_target_adv < Q_star:
                    Q_star = q_target_adv
                    s_adv = si
                else:
                    pass
            action = int(model(s_adv).data.max(1)[1].cpu().numpy())
        else:
            action = random.randrange(num_actions)
        return torch.IntTensor([[action]]) 
   
    def get_state_value(model, obs, num_actions, this_action=None):
        obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255.0
        values = model(Variable(obs, volatile=True)).data.cpu().numpy()
        values = values.reshape((num_actions,))
        state_value = np.max(values)
        if this_action is not None:
            state_value = values[int(this_action)]
        return state_value
 
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs                 = env.reset()
    LOG_EVERY_N_STEPS        = 10000

    episode_rewards = []
    episode_reward = 0.0
    num_epi = 0

    if args.load_old_q_value == True:
        try:
            log_file = open(os.path.join(args.save_path, args.log_name), 'rb').readlines()
            last_line = str(log_file[-1])
            num_epi = int(last_line.split(' ')[1])
        except:
            pass
    
    epi_len = 0
    for t in itertools.count():
        with global_lock:
            global_step.value += 1
        t = global_step.value
        if args.without_dla == False:
            last_obs = cv2.resize(last_obs, (256,256))
        ret = replay_buffer.store_frame( last_obs )
        obs = replay_buffer.encode_recent_observation()
        if args.with_adv == True:
            action = int(select_attack_action(q_value, target_q_value, obs, t,\
                        n = args.n_iter, alpha=args.alpha, beta=args.beta)[0,0])
        else:
            action = int(select_epilson_greedy_action(q_value, obs, global_step.value)[0, 0])

        last_obs, reward, done, info = env.step(action)
        epi_len += 1
        pos = np.array((info['pos'][0], info['pos'][1], info['pos'][2]))
        dest = np.array((935.30, 392.00, 1.7580))
        if np.sqrt(np.sum((pos-dest)**2.0)) <= 20 and args.early_stop:
            done = True
        done = done or reward <=-2.5 or epi_len >= 1000
        if reward > -2.5:
            episode_reward += reward
    
        this_value = 0#get_state_value(target_q_value, obs, 9)
        this_q_value = 0#get_state_value(target_q_value, obs, 9, action)

        with open(os.path.join(args.save_path, 'log_value.txt'), 'a') as fi:
            fi.write('step '+str(t)+' action '+str(action)+' reward '+str(reward)+' value '+str(this_value)+' qvalue '+str(this_q_value)+' pos '+str(info['pos'][0])+' '+str(info['pos'][1])+' '+str(info['pos'][2])+' done '+str(int(done))+'\n')

        if done:
            epi_len = 0
            last_obs = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            num_epi += 1

        replay_buffer.store_effect(ret, action, reward, done)
 
        if (global_step.value > learning_starts and
                global_step.value % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)

            obs_t_batch     = Variable(torch.from_numpy(obs_t_batch).type(dtype) / 255.0)
            act_batch       = Variable(torch.from_numpy(act_batch).long())
            rew_batch       = Variable(torch.from_numpy(rew_batch))
            obs_tp1_batch   = Variable(torch.from_numpy(obs_tp1_batch).type(dtype) / 255.0)
            done_mask       = Variable(torch.from_numpy(done_mask)).type(dtype)

            if use_cuda == True:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
            q_a_values = q_value(obs_t_batch).gather(1, act_batch.unsqueeze(1))
            q_a_values_tp1 = target_q_value(obs_tp1_batch).detach().max(1)[0]
            target_values = rew_batch + (gamma * (1-done_mask) * q_a_values_tp1)
            loss = ((target_values.view(q_a_values.size()) - q_a_values)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_param_updates += 1

            # update the target network
            if num_param_updates % target_update_freq==0:
                target_q_value.load_state_dict(q_value.state_dict())
                if int(int(global_step.value)/1000)*1000 % 30000 == 0:
                    if os.path.isdir(os.path.join(args.save_path, 'models', str(global_step.value))) == False:
                        os.mkdir(args.save_path+'/models/'+str(global_step.value))
                    torch.save(target_q_value.state_dict(), args.save_path+'/models/'+str(global_step.value)+'/model_'+str(0)+'.pt')
 
        ### 4. Log progress
        mean_episode_reward = -12
        var_reward = 0.0
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            var_reward = np.std(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and t > 0:
            print("Timestep %d number of episode %d" % (global_step.value, num_epi))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("var of reward %f" % var_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(global_step.value))
        if done:
            with open(os.path.join(args.save_path, args.log_name), 'a') as fi:
                fi.write('timestep '+str(global_step.value)+\
                        ' mean reward '+str(mean_episode_reward)+\
                        ' var reward '+str(var_reward)+'\n')
            sys.stdout.flush()
