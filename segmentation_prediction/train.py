import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from utils import clear_visualization_dir, reset_env, naive_driver, init_variables, init_criteria, draw_from_pred

from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt

def sample_action(args, model, predictor, further):
    return np.random.randint(args.num_actions)

def sample_one_step(args, true_obs, inputs, targets, target_pos, target_angle, actions, env, model, up, predictor, further):
    exploration = args.exploration_decay ** args.epoch

    action = sample_action(args, model, predictor, further) if np.random.random() > exploration else naive_driver(env.get_info())
    obs, reward, done, info = env.step(action)
    obs = (obs.transpose(2, 0, 1) - args.obs_avg) / args.obs_std
    true_obs[:] = np.concatenate((true_obs[args.frame_len:], obs), axis = 0)

    inputs[args.step, args.batch_id] = torch.from_numpy(true_obs)
    targets[args.step, args.batch_id] = torch.from_numpy(env.get_segmentation().astype(np.uint8))
    target_pos[args.step, args.batch_id] = info['trackPos'] * args.pos_rescale
    target_angle[args.step, args.batch_id] = info['angle'] * args.angle_rescale
    if args.step > 0:
        actions[args.step - 1, args.batch_id] = action
    
    if done or reward <= -2.5:
        true_obs[:] = reset_env(args, env)
    return done or reward <= -2.5


def collect_data(args, true_obs, inputs, targets, target_pos, target_angle, actions, env, model, up, predictor, further):
    args.step = 0
    while args.step <= args.num_steps:
        terminal = sample_one_step(args, true_obs, inputs, targets, target_pos, target_angle, actions, env, model, up, predictor, further)
        args.step += 1
        if terminal and args.step < args.num_steps:
            args.step = 0

def train_data(args, inputs, outputs, prediction, feature_map, output_pos, output_angle, targets, target_pos, target_angle, actions, model, up, predictor, further, optimizer, criterion):
    LOSS = np.zeros((args.num_steps + 1, 4))
    dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    weight = 1
    loss = 0
    optimizer.zero_grad()

    _feature_map = model(inputs[0])
    feature_map[0] = _feature_map
    _prediction = up(_feature_map)
    prediction[0] = _prediction
    _output_pos, _output_angle = further(_feature_map)
    output_pos[0], output_angle[0] = _output_pos, _output_angle

    loss0 = criterion['CE'](_prediction, targets[0])
    loss2 = criterion['L2'](_output_pos, target_pos[0]) / 1000
    loss3 = criterion['L2'](_output_angle, target_angle[0]) / 1000
    loss = loss0 + loss2 + loss3
    LOSS[0, 0] = loss0.data.cpu().numpy()
    LOSS[0, 2] = loss2.data.cpu().numpy()
    LOSS[0, 3] = loss3.data.cpu().numpy()

    for i in range(1, args.num_steps + 1):
        weight *= args.loss_decay
        _outputs = up(model(inputs[i]))
        _feature_map = predictor(_feature_map, actions[i - 1])
        feature_map[i] = _feature_map
        _prediction = up(_feature_map)
        prediction[i] = _prediction
        _output_pos, _output_angle = further(_feature_map)
        output_pos[i], output_angle[i] = _output_pos, _output_angle
        loss0 = criterion['CE'](_outputs, targets[i])
        loss1 = criterion['CE'](_prediction, targets[i]) # torch.argmax(_outputs, dim = 1).type(dtype)
        loss2 = criterion['L2'](_output_pos, target_pos[i]) / 1000
        loss3 = criterion['L2'](_output_angle, target_angle[i]) / 1000
        loss += weight * (loss0 + loss1 + loss2 + loss3)
        LOSS[i, 0] = loss0.data.cpu().numpy()
        LOSS[i, 1] = loss2.data.cpu().numpy()
        LOSS[i, 2] = loss2.data.cpu().numpy()
        LOSS[i, 3] = loss3.data.cpu().numpy()
    losses = loss.data.cpu().numpy()
    loss.backward()
    optimizer.step()

    return LOSS, losses

def visualize_data(args, inputs, prediction, output_pos, output_angle, targets, target_pos, target_angle, actions):
    print('Visualizing data.')
    clear_visualization_dir(args)
    batch_id = np.random.randint(args.batch_size)
    pred = torch.argmax(prediction[:, batch_id, :, :, :], dim = 1)
    all_obs = torch.round(inputs.data[:, batch_id, -3:, :, :] * args.obs_std + args.obs_avg)
    if torch.cuda.is_available():
        all_obs = all_obs.cpu()
    all_obs = all_obs.numpy().transpose(0, 2, 3, 1)
    for i in range(args.num_steps + 1):
        imsave('pred_images/%d.png' % i, np.concatenate((all_obs[i], draw_from_pred(targets[i, batch_id]), draw_from_pred(pred[i])), axis = 1))


def train(args, model, up, predictor, further, env):
    model.train()
    up.train()
    predictor.train()
    further.train()

    inputs, outputs, prediction, feature_map, output_pos, output_angle, targets, target_pos, target_angle, actions = init_variables(args)
    criterion = init_criteria()
    optimizer = torch.optim.Adam(list(model.parameters()) \
                               + list(up.parameters()) \
                               + list(predictor.parameters()) \
                               + list(further.parameters()),
                                 args.lr,
                                 amsgrad = True)
    true_obs = reset_env(args, env)

    while True:
        for args.batch_id in range(args.batch_size):
            collect_data(args, true_obs, inputs, targets, target_pos, target_angle, actions, env, model, up, predictor, further)
        LOSS, losses = train_data(args, inputs, outputs, prediction, feature_map, output_pos, output_angle, targets, target_pos, target_angle, actions, model, up, predictor, further, optimizer, criterion)
        args.epoch += 1

        print(LOSS)
        print('Iteration %d, mean loss %f' % (args.epoch, losses))
        with open('pred_log.txt', 'a') as f:
            f.write('%s\nIteration %d, mean loss %f\n' % (str(LOSS), args.epoch, losses))

        if args.epoch % 20 == 0:
            visualize_data(args, inputs, prediction, output_pos, output_angle, targets, target_pos, target_angle, actions)
        
        if args.epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model_epoch%d.dat' % args.epoch))
            torch.save(up.state_dict(), os.path.join(args.model_dir, 'up_epoch%d.dat' % args.epoch))
            torch.save(predictor.state_dict(), os.path.join(args.model_dir, 'predictor_epoch%d.dat' % args.epoch))
            torch.save(further.state_dict(), os.path.join(args.model_dir, 'further_epoch%d.dat' % args.epoch))

