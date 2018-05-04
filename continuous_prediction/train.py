import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from replay_buffer import replay_buffer
from utils import clear_visualization_dir, reset_env, naive_driver, init_criteria, draw_from_pred, from_variable_to_numpy, sample_action, sample_continuous_action, convert_state_dict

from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt

def sample_one_step(args, true_obs, env, model, posxyz):
    exploration = args.exploration_decay ** args.epoch
    data = {'obs': true_obs}

    if args.continuous:
        action, dqn_action = sample_continuous_action(args, true_obs, model)
        data['dqn_action'] = dqn_action
    else:
        action = sample_action(args, true_obs, model)
    data['action'] = action

    obs, reward, done, info = env.step(action)
    with open('env_log.txt', 'a') as f:
        f.write('reward = %0.4f\n' % reward)
    obs = (obs.transpose(2, 0, 1) - args.obs_avg) / args.obs_std
    if args.frame_len > 1:
        true_obs = np.concatenate((true_obs[3:], obs), axis = 0)
    else:
        true_obs = obs

    if args.use_dqn:
        data['reward'] = reward
    if args.use_seg:
        data['target_seg'] = env.get_segmentation().astype(np.uint8)

    data['target_coll'] = int(reward <= -2.5 or abs(info['trackPos']) > 7)
    data['target_off'] = int(info['trackPos'] >= 3 or info['trackPos'] <= -1)
    if args.use_center_dist:
        data['target_dist'] = abs(info['trackPos'])
    else:
        data['target_dist'] = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])) - np.abs(info['trackPos']) / 9.0)

    if args.use_xyz:
        data['target_xyz'] = np.array(info['pos']) - posxyz
        posxyz = np.array(info['pos'])
    
    done = done or reward <= -2.5 # or abs(info['trackPos']) > args.coll_rescale or abs(info['off']) > args.off_rescale
    if done:
        with open('env_log.txt', 'a') as f:
            f.write('done')
        true_obs, posxyz = reset_env(args, env)
    data['done'] = int(done)
    return data, true_obs, posxyz

def train_model(args, train_data, model, optimizer): # need updating
    output = model(obs, actions)
    weight = Variable(args.loss_decay ** torch.arange(args.num_steps), requires_grad = False).view(args.num_steps, 1)

    criterion = init_criteria()
    
    coll_loss = torch.sum(criterion['BCE'](output['output_coll'], train_data['target_coll']) * weight)
    off_loss = torch.sum(criterion['BCE'](output['output_off'], train_data['target_off']) * weight)
    dist_loss = torch.sum(criterion['L2'](output['output_dist'], train_data['target_dist']) * weight)
    loss = coll_loss + off_loss + dist_loss
    if args.use_xyz:
        loss += torch.sum(criterion['L2'](output['output_xyz'], train_data['target_xyz']) * weight)
    if args.use_seg:
        loss += torch.sum(criterion['CE'](output['output_seg'], train_data['target_seg']) * weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    LOSS = from_variable_to_numpy(loss + train_dqn(args, train_data, model.module.dqn, optimizer))

    return LOSS, output

def train_dqn(args, train_data, model, optimizer):
    q_a_values = dqn.dqn(train_data['obs']).gather(1, train_data['dqn_action'].unsqueeze(1))
    with torch.no_grad():
        q_a_values_tp1 = dqn.target_Q(train_data['obs']).detach().max(1)[0]
    target_values = train_data['reward'] + (0.99 * q_a_values_tp1)
    dqn_loss = ((target_values.view(q_a_values.size()) - q_a_values) ** 2).mean()
    optimizer.zero_grad()
    dqn_loss.backward()
    optimizer.step()
    if args.epoch % 20 == 0:
        dqn.target_Q.load_state_dict(convert_state_dict(dqn.dqn.state_dict()))

def visualize_data(args, output, train_data):
    clear_visualization_dir(args)
    if args.use_seg:
        batch_id = np.random.randint(args.batch_size)
        pred = torch.argmax(output['output_seg'][:, batch_id, :, :, :], dim = 1)
        all_obs = torch.round(train_data['obs'].data[:, batch_id, -3:, :, :] * args.obs_std + args.obs_avg)
        if torch.cuda.is_available():
            all_obs = all_obs.cpu()
        all_obs = all_obs.numpy().transpose(0, 2, 3, 1)
        for i in range(args.num_steps + 1):
            imsave(os.path.join(args.visualize_dir, '%d.png' % i), np.concatenate((all_obs[i], draw_from_pred(train_data['target_seg'][i, batch_id]), draw_from_pred(pred[i])), axis = 1))

    with open(os.path.join(args.visualize_dir, 'cmp.txt'), 'w') as f:
        f.write('target coll:\n%s\n' % str(from_variable_to_numpy(train_data['target_coll'])))
        f.write('predicted coll:\n%s\n\n' % str(from_variable_to_numpy(output['output_coll'])))
        f.write('target off:\n%s\n' % str(from_variable_to_numpy(train_data['target_off'])))
        f.write('predicted off:\n%s\n' % str(from_variable_to_numpy(output['output_off'])))
        f.write('target dist:\n%s\n' % str(from_variable_to_numpy(train_data['target_dist'])))
        f.write('predicted dist:\n%s\n' % str(from_variable_to_numpy(output['output_dist'])))
    

def train(args, model, env):
    model.train()
    model.module.dqn.target_Q.eval()

    buffer = replay_buffer(args)
    optimizer = torch.optim.Adam(list(model.parameters()), args.lr, amsgrad = True)
    true_obs, posxyz = reset_env(args, env)

    while True:
        for i in range(args.collect_steps):
            data, true_obs, posxyz = sample_one_step(args, true_obs, env, model, posxyz)
            buffer.store(data)
        train_data = buffer.sample()
        LOSS, loss = train_model(args, train_data, model, optimizer)
        args.epoch += 1

        print(LOSS)
        print('Iteration %d, mean loss %f' % (args.epoch, loss))
        with open('pred_log.txt', 'a') as f:
            f.write('%s\nIteration %d, mean loss %f\n' % (str(LOSS), args.epoch, loss))

        if args.epoch % 10 == 0:
            print('Visualizing data.')
            visualize_data(args, train_data)
        
        if args.epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model_epoch%d.dat' % args.epoch))
    #     break
    # env.close()