import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from utils import clear_visualization_dir, reset_env, naive_driver, init_variables, init_criteria, draw_from_pred, from_variable_to_numpy, sample_action

from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt

def sample_one_step(args, true_obs, inputs, targets, target_pos, target_angle, actions, env, model, up, predictor, further):
    exploration = args.exploration_decay ** args.epoch

    action = sample_action(args, true_obs, model, predictor, further) if np.random.random() > exploration else naive_driver(args, env.get_info())
    obs, reward, done, info = env.step(action)
    obs = (obs.transpose(2, 0, 1) - args.obs_avg) / args.obs_std
    true_obs[:] = np.concatenate((true_obs[args.frame_len:], obs), axis = 0)

    inputs[args.step, args.batch_id] = torch.from_numpy(true_obs)
    targets[args.step, args.batch_id] = torch.from_numpy(env.get_segmentation().astype(np.uint8))
    target_pos[args.step, args.batch_id] = info['trackPos'] / args.pos_rescale
    target_angle[args.step, args.batch_id] = info['angle'] / args.angle_rescale
    if args.step > 0:
        actions[args.step - 1, args.batch_id] = action
    
    done = done or reward <= -2.5 or abs(info['trackPos']) > args.pos_rescale or abs(info['angle']) > args.angle_rescale
    if done:
        true_obs[:] = reset_env(args, env)
    return done


def collect_data(args, true_obs, inputs, targets, target_pos, target_angle, actions, env, model, up, predictor, further):
    args.step = 0
    while args.step <= args.num_steps:
        terminal = sample_one_step(args, true_obs, inputs, targets, target_pos, target_angle, actions, env, model, up, predictor, further)
        args.step += 1
        if terminal and args.step < args.num_steps:
            args.step = 0

def train_data(args, inputs, prediction, output_pos, output_angle, targets, target_pos, target_angle, actions, model, up, predictor, further, optimizer, criterion):
    LOSS = np.zeros((args.num_steps + 1, 4))
    weight = 1

    _feature_map = model(inputs[0])
    _prediction = up(_feature_map)
    prediction[0] = _prediction
    _output_pos, _output_angle = further(_feature_map)
    output_pos[0], output_angle[0] = _output_pos, _output_angle

    loss0 = criterion['CE'](_prediction, targets[0])
    loss2 = criterion['L2'](_output_pos, target_pos[0]) / 10
    loss3 = criterion['L2'](_output_angle, target_angle[0]) / 10
    loss = loss0 + loss2 + loss3
    LOSS[0, 0] = from_variable_to_numpy(loss0)
    LOSS[0, 2] = from_variable_to_numpy(loss2)
    LOSS[0, 3] = from_variable_to_numpy(loss3)

    for i in range(1, args.num_steps + 1):
        weight *= args.loss_decay
        _outputs = up(model(inputs[i]))
        _feature_map = predictor(_feature_map, actions[i - 1])
        _prediction = up(_feature_map)
        prediction[i] = _prediction
        _output_pos, _output_angle = further(_feature_map)
        output_pos[i], output_angle[i] = _output_pos, _output_angle
        loss0 = criterion['CE'](_outputs, targets[i])
        loss1 = criterion['CE'](_prediction, targets[i]) / 10
        loss2 = criterion['L2'](_output_pos, target_pos[i]) / 10
        loss3 = criterion['L2'](_output_angle, target_angle[i]) / 10
        loss += loss0 + weight * (loss1 + loss2 + loss3)
        LOSS[i, 0] = from_variable_to_numpy(loss0)
        LOSS[i, 1] = from_variable_to_numpy(loss1)
        LOSS[i, 2] = from_variable_to_numpy(loss2)
        LOSS[i, 3] = from_variable_to_numpy(loss3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return LOSS, from_variable_to_numpy(loss)

def visualize_data(args, inputs, prediction, output_pos, output_angle, targets, target_pos, target_angle, actions):
    print('Visualizing data.')
    clear_visualization_dir(args)
    batch_id = np.random.randint(args.batch_size)
    pred = torch.argmax(prediction[:, batch_id, :, :, :], dim = 1)
    all_obs = torch.round(inputs.data[:, batch_id, -3:, :, :] * args.obs_std + args.obs_avg)
    if torch.cuda.is_available():
        all_obs = all_obs.cpu()
    all_obs = all_obs.numpy().transpose(0, 2, 3, 1)
    with open(os.path.join(args.visualize_dir, 'cmp.txt'), 'w') as f:
        f.write('target pos:\n%s\n' % str(from_variable_to_numpy(target_pos)))
        f.write('predicted pos:\n%s\n\n' % str(from_variable_to_numpy(output_pos)))
        f.write('target angle:\n%s\n' % str(from_variable_to_numpy(target_angle)))
        f.write('predicted angle:\n%s\n' % str(from_variable_to_numpy(output_angle)))
    for i in range(args.num_steps + 1):
        imsave(os.path.join(args.visualize_dir, '%d.png' % i), np.concatenate((all_obs[i], draw_from_pred(targets[i, batch_id]), draw_from_pred(pred[i])), axis = 1))


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
        LOSS, losses = train_data(args, inputs, prediction, output_pos, output_angle, targets, target_pos, target_angle, actions, model, up, predictor, further, optimizer, criterion)
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
    #     break
    # env.close()