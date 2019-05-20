from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle as pkl
import os
import time
from sklearn.metrics import confusion_matrix
import copy
import cv2
import logging
from utils.eval_segm import mean_IU, mean_accuracy, pixel_accuracy, frequency_weighted_IU


def setup_logger(logger_name, log_file, level=logging.INFO, resume=False):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a' if resume else 'w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        indices = [e[0] for e in endpoints]
        assert indices == sorted(indices)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        assert self._outside_value is not None
        return self._outside_value


def generate_guide_grid(bin_divide, lb=-1.0, ub=1.0):
    grids = np.meshgrid(*map(lambda x: (np.arange(x) + 0.5) / x * (ub - lb) + lb, bin_divide))
    return np.concatenate(list(map(lambda x: x.reshape(-1, 1), grids)), axis=-1)


def softmax(x, axis=1):
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis=axis))
    return e_x / np.expand_dims(np.sum(e_x, axis=axis), axis=axis)


def log_frame(obs, action, video_folder, video=None):
    if video is not None:
        video.write(obs)
    with open(os.path.join(video_folder, 'actions.txt'), 'a') as f:
        f.write('time %0.2f action %0.4f %0.4f\n' % (
            time.time(),
            action[0],
            action[1]
        ))


def color_text(text, color):
    color = color.lower()
    if color == 'red':
        prefix = '\033[1;31m'
    elif color == 'green':
        prefix = '\033[1;32m'
    return prefix + text + '\033[0m'


def train_guide_action(args, net, spc_buffer, guides):
    assert args.use_guidance
    if spc_buffer.can_sample_guide(args.batch_size):
        obs, guide_action = spc_buffer.sample_guide(args.batch_size)
        q = net(obs, function='guide_action')
        loss = nn.CrossEntropyLoss()(q, guide_action)
        if args.verbose:
            visualize_guide_action(args, obs, q, guides, guide_action)
        print('Guidance loss: %0.4f' % loss.data.cpu().numpy())
        return loss
    else:
        print(color_text('Insufficient expert data for imitation learning.', 'red'))
        return 0.0


def generate_episode(args, mean, lb=-1.0, ub=1.0):
    res = []
    full_range = ub - lb
    for i in range(args.pred_step):
        res.append(np.array(mean) + np.array(list(map(lambda x: np.random.uniform(low=-full_range / 2.0 / x, high=full_range / 2.0 / x), args.bin_divide))))
    res = list(map(lambda x: x.reshape(1, -1), res))
    return np.concatenate(res, axis=0)


def generate_action(args, p, size, guides, lb=-1.0, ub=1.0):
    res = []
    for _ in range(size):
        c = np.random.choice(range(len(p)), p=p)
        res.append(np.expand_dims(generate_episode(args, guides[c], lb, ub), axis=0))
    return np.concatenate(res, axis=0)


def get_guide_action(bin_divide, action, lb=-1.0, ub=1.0):
    _bin_divide = np.array(bin_divide)
    action = ((action - lb) / (ub - lb) * _bin_divide).astype(np.uint8)
    weight = np.array(list(map(lambda x: np.prod(_bin_divide[:x]), range(len(bin_divide)))))
    return int(np.sum(action * weight))


def visualize_guide_action(args, data, outputs, guides, label):
    if not os.path.isdir('visualize/guidance'):
        os.makedirs('visualize/guidance')
    _outputs = F.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    label = label.data.cpu().numpy()
    for i in range(data.shape[0]):
        obs = data[i].data.cpu().numpy().transpose(1, 2, 0)
        obs = cv2.cvtColor((obs * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        action = guides[int(outputs[i])]
        obs = draw_guide(args, obs, 150, 66, 45, _outputs[i].data.cpu().numpy().reshape(-1))
        obs = draw_action(obs, 150, 66, 45, 1, np.array(action))
        gt_action = guides[int(label[i])]
        obs = draw_action(obs, 150, 190, 45, 1, np.array(gt_action))
        cv2.imwrite(os.path.join('visualize', 'guidance', 'guidance_%d.png' % i), obs)


def draw_guide(args, fig, x, y, l, p):
    square = np.ones((args.bin_divide[0]*6+1, args.bin_divide[1]*6+1, 3), dtype=np.uint8) * 128
    p = p * 255 * 10
    for i in range(args.bin_divide[1]):
        for j in range(args.bin_divide[0]):
            square[i*6+1:i*6+6, j*6+1:j*6+6, :] = p[j*args.bin_divide[1]+i]
    square = np.flip(square, axis=0)
    square = cv2.resize(square, (2*l, 2*l))
    fig[x-l:x+l, y-l:y+l, :] = square
    return fig


def draw_action(fig, x, y, l, w, action):
    fig[x-l:x+l, y-w:y+w] = 0
    fig[x-w:x+w, y-l:y+l] = 0
    t = int(abs(action[0]) * l)
    if action[0] > 0:
        fig[x-t:x, y-3*w:y+3*w] = np.array([36, 28, 237])
    else:
        fig[x:x+t, y-3*w:y+3*w] = np.array([36, 28, 237])
    t = int(abs(action[1]) * l)
    if action[1] > 0:
        fig[x-3*w:x+3*w, y:y+t] = np.array([14, 201, 255])
    else:
        fig[x-3*w:x+3*w, y-t:y] = np.array([14, 201, 255])
    return fig


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1: 4])
        fan_out = np.prod(weight_shape[2: 4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


def write_log(args, file, func, output, target):
    res = []
    for i in range(args.pred_step + 1):
        tmp = 0
        for j in range(args.batch_size):
            tmp += func(output[j, i, ...], target[j, i, ...])
        res.append(tmp * 100 / args.batch_size)
    with open(file, 'a') as f:
        for i in range(args.pred_step + 1):
            f.write('%0.3f ' % res[i])
        f.write('\n')


def train_model(args, net, spc_buffer):
    target = spc_buffer.sample(args.batch_size)

    for key in target.keys():
        target[key] = Variable(torch.from_numpy(target[key]).float(), requires_grad=False)

    target['obs_batch'] = target['obs_batch'] / 255.0

    if args.no_supervision:
        target['nx_obs_batch'] = target['nx_obs_batch'] / 255.0
        with torch.no_grad():
            nximg_enc = net(target['nx_obs_batch'], get_feature=True, next_obs=True).detach()
        nximg_enc = nximg_enc[:, :, -args.classes:, :, :]
    else:
        target['seg_batch'] = target['seg_batch'].long()

    if torch.cuda.is_available():
        for key in target.keys():
            target[key] = target[key].cuda()

    output = net(target['obs_batch'], target['act_batch'], action_var=target['prev_action'])

    loss = 0

    weight = (args.time_decay ** np.arange(args.pred_step)).reshape((1, args.pred_step, 1))
    weight = Variable(torch.from_numpy(weight).float().cuda(), requires_grad=False).repeat(args.batch_size, 1, 1)

    if args.use_collision:
        acc = get_accuracy(target['coll_batch'].view(-1).data.cpu().numpy(), torch.max(output['coll_prob'].view(-1, 2), -1)[1].data.cpu().numpy())
        print('Collision accuracy: {0:.2f}%'.format(acc))

        if args.verbose:
            coll_np = torch.argmax(output['coll_prob'].view(args.batch_size, args.pred_step, 2), dim=2).data.cpu().numpy()
            coll_target_np = target['coll_batch'].view(args.batch_size, args.pred_step).data.cpu().numpy()
            colls = []
            for i in range(args.pred_step):
                colls.append(get_accuracy(coll_target_np[:, i], coll_np[:, i]))
            with open(os.path.join(args.save_path, 'coll_log.txt'), 'a') as f:
                for i in range(args.pred_step):
                    f.write('%0.3f ' % colls[i])
                f.write('\n')

        coll_ls = nn.CrossEntropyLoss()(output['coll_prob'].view(-1, 2), target['coll_batch'].view(-1).long())
        loss += coll_ls
        print('Collision loss:', coll_ls.data.cpu().numpy())

    if args.use_offroad:
        acc = get_accuracy(target['off_batch'].view(-1).data.cpu().numpy(), torch.max(output['offroad_prob'].view(-1, 2), -1)[1].data.cpu().numpy())
        print('Offroad accuracy: {0:.2f}%'.format(acc))

        if args.verbose:
            off_np = torch.argmax(output['offroad_prob'].view(args.batch_size, args.pred_step, 2), dim=2).data.cpu().numpy()
            off_target_np = target['off_batch'].view(args.batch_size, args.pred_step).data.cpu().numpy()
            offs = []
            for i in range(args.pred_step):
                offs.append(get_accuracy(off_target_np[:, i], off_np[:, i]))
            with open(os.path.join(args.save_path, 'off_log.txt'), 'a') as f:
                for i in range(args.pred_step):
                    f.write('%0.3f ' % offs[i])
                f.write('\n')

        offroad_ls = nn.CrossEntropyLoss()(output['offroad_prob'].view(-1, 2), target['off_batch'].view(-1).long())
        loss += offroad_ls
        print('Offroad loss:', offroad_ls.data.cpu().numpy())

    if args.use_speed:
        speed_loss = torch.sqrt(nn.MSELoss()(output['speed'], target['sp_batch'][:, 1:].unsqueeze(dim=2)))
        loss += 0.01 * speed_loss
        print('speed ls', speed_loss.data.cpu().numpy())

    if args.no_supervision:
        pred = output['seg_pred'][:, 1:, ...].contiguous().view(args.batch_size * args.pred_step, args.classes * args.frame_height * args.frame_width)
        nximg_enc = nximg_enc.contiguous().view(args.batch_size * args.pred_step, args.classes * args.frame_height * args.frame_width)
        pred_ls = torch.sum(nn.KLDivLoss(reduce=False)(pred, nximg_enc), dim=-1)
        pred_ls = torch.sum(pred_ls.contiguous().view(args.batch_size, args.pred_step, 1) * weight) / args.batch_size / args.frame_height / args.frame_width
    else:
        output['seg_pred'] = output['seg_pred'].view(args.batch_size * (args.pred_step + 1), args.classes, 256, 256)
        target['seg_batch'] = target['seg_batch'].view(args.batch_size * (args.pred_step + 1), 256, 256)
        pred_ls = nn.NLLLoss()(output['seg_pred'], target['seg_batch'])
        seg_np = torch.argmax(output['seg_pred'].view(args.batch_size, args.pred_step + 1, args.classes, 256, 256), dim=2).data.cpu().numpy()
        target_np = target['seg_batch'].view(args.batch_size, args.pred_step + 1, 256, 256).data.cpu().numpy()

        if args.verbose:
            write_log(args, os.path.join(args.save_path, 'mean_IU.txt'), mean_IU, seg_np, target_np)
            write_log(args, os.path.join(args.save_path, 'mean_acc.txt'), mean_accuracy, seg_np, target_np)
            write_log(args, os.path.join(args.save_path, 'pixel_acc.txt'), pixel_accuracy, seg_np, target_np)
            write_log(args, os.path.join(args.save_path, 'freq_IU.txt'), frequency_weighted_IU, seg_np, target_np)

    print('Segmentation loss:', pred_ls.data.cpu().numpy())  # nan here!
    loss += pred_ls

    if args.verbose:
        visualize(args, target, output)
    return loss


def draw_from_pred_torcs(pred):
    illustration = np.zeros((256, 256, 3)).astype(np.uint8)
    illustration[:, :, 0] = 255
    illustration[pred == 1] = np.array([0, 255, 0])
    illustration[pred == 2] = np.array([0, 0, 0])
    illustration[pred == 3] = np.array([0, 0, 255])
    return illustration


def draw_from_pred_carla(array):
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }

    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


def draw_from_pred_gta(array):
    classes = {
        0: [0, 0, 0],
        1: [255, 255, 255],
        2: [255, 0, 0],
        3: [0, 255, 0],
        4: [0, 0, 255],
        5: [255, 255, 0],
        6: [0, 255, 255],
        7: [255, 0, 255],
        8: [192, 192, 192],
        9: [128, 128, 128],
        10: [128, 0, 0],
        11: [128, 128, 0],
        12: [0, 128, 0],
        13: [128, 0, 128],
        14: [0, 128, 128],
        15: [0, 0, 128],
        16: [139, 0, 0],
        17: [165, 42, 42],
        18: [178, 34, 34]
    }

    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


def draw_from_pred(args, array):
    if 'torcs' in args.env :
        return draw_from_pred_torcs(array)
    elif 'carla' in args.env:
        return draw_from_pred_carla(array)
    elif 'gta' in args.env:
        return draw_from_pred_gta(array)


def visualize(args, target, output):
    if not os.path.isdir('visualize'):
        os.mkdir('visualize')

    batch_id = np.random.randint(args.batch_size)
    observation = (from_variable_to_numpy(target['obs_batch'][batch_id, :, -3:, :, :]) * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)
    target['seg_batch'] = target['seg_batch'].view(args.batch_size, args.pred_step + 1, 256, 256)
    segmentation = from_variable_to_numpy(target['seg_batch'][batch_id])
    output['seg_pred'] = output['seg_pred'].view(args.batch_size, args.pred_step + 1, args.classes, 256, 256)
    _, prediction = torch.max(output['seg_pred'][batch_id], 1)
    prediction = from_variable_to_numpy(prediction)
    for i in range(args.pred_step):
        cv2.imwrite('visualize/%d.png' % i, np.concatenate([cv2.cvtColor(observation[i], cv2.COLOR_RGB2BGR), draw_from_pred(args, segmentation[i]), draw_from_pred(args, prediction[i])], 1))

    with open(os.path.join(args.save_path, 'report.txt'), 'a') as f:
        if args.use_collision:
            f.write('target collision:\n')
            f.write(str(from_variable_to_numpy(target['coll_batch'][batch_id])) + '\n')
            f.write('output collision:\n')
            f.write(str(from_variable_to_numpy(output['coll_prob'][batch_id])) + '\n')

        if args.use_offroad:
            f.write('target offroad:\n')
            f.write(str(from_variable_to_numpy(target['off_batch'][batch_id])) + '\n')
            f.write('output offroad:\n')
            f.write(str(from_variable_to_numpy(output['offroad_prob'][batch_id])) + '\n')

        if args.use_speed:
            f.write('target speed:\n')
            f.write(str(from_variable_to_numpy(target['sp_batch'][batch_id, :-1])) + '\n')
            f.write('output speed:\n')
            f.write(str(from_variable_to_numpy(output['speed'][batch_id])) + '\n')


def setup_dirs(args):
    init_dirs([args.save_path,
               os.path.join(args.save_path, 'model'),
               os.path.join(args.save_path, 'optimizer'),
               os.path.join(args.save_path, 'cnfmat')])


def init_dirs(dir_list):
    for path in dir_list:
        make_dir(path)


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def load_model(path, net, resume=True):
    if resume:
        file_list = sorted(os.listdir(os.path.join(path, 'model')))
        if len(file_list) == 0:
            print('No model to resume!')
            epoch = 0
        else:
            model_path = file_list[-2]
            epoch = pkl.load(open(os.path.join(path, 'epoch.pkl'), 'rb'))
            print('Loading model from', os.path.join(path, 'model', model_path))
            state_dict = torch.load(os.path.join(path, 'model', model_path))
            net.load_state_dict(state_dict, strict=False)
    else:
        epoch = 0

    return net, epoch


def sample_action(args, p, net, imgs, guides, action_var=None, testing=False):
    imgs = copy.deepcopy(imgs) / 255.0
    batch_size, c, w, h = int(imgs.size()[0]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
    imgs = imgs.view(batch_size, 1, c, w, h)

    imgs = imgs.repeat(25, 1, 1, 1, 1)
    action_var = action_var.repeat(25, 1, 1)

    if args.use_guidance:
        action = generate_action(args, p, 25, guides)
    else:
        action = guides.reshape(25, 1, 2).repeat(10, axis=1)
    this_action0 = copy.deepcopy(action)
    this_action = Variable(torch.from_numpy(action).cuda().float(), requires_grad=False)

    with torch.no_grad():
        start_time = time.time()
        loss = get_action_loss(args, net, imgs, this_action, action_var, None, None, None).data.cpu().numpy()
        print('Sampling takes %0.2f seconds.' % (time.time() - start_time))

    idx = np.argmin(loss)
    res = this_action0[idx, :, :]
    if not testing:
        res = res[0]
    return res


def from_variable_to_numpy(x):
    x = x.data
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.numpy()
    return x


def tile_single(x, action):
    batch_size, c, w, h = x.size()
    assert action.size(0) == batch_size
    action = action.view(action.size(0), -1, 1, 1).repeat(1, 1, w, h)
    return torch.cat([x, action], dim=1)


def tile(x, action):
    return list(map(lambda t: tile_single(t, action), x))


def tile_first(x, action):
    for i in range(len(x) - 1):
        x[i] = tile(x[i], action[:, i, :].float())
    return x


def get_action_loss(args, net, imgs, actions, action_var=None, target=None, hidden=None, cell=None, gpu=0):
    batch_size = int(imgs.size()[0])

    weight = (args.time_decay ** np.arange(args.pred_step)).reshape((1, args.pred_step, 1))
    weight = Variable(torch.from_numpy(weight).float().cuda()).repeat(batch_size, 1, 1)
    output = net(imgs, actions, hidden=hidden, cell=cell, training=False, action_var=action_var)

    loss = 0

    if args.sample_with_collision:
        output['coll_prob'] = F.softmax(output['coll_prob'], -1)
        coll_ls = -torch.round(output['coll_prob'][:, :, 0]) * output['speed'].view(-1, args.pred_step) + torch.round(output['coll_prob'][:, :, 1]) * args.speed_threshold
        coll_ls = (coll_ls.view(-1, args.pred_step, 1) * weight).sum(-1).sum(-1)
        loss += coll_ls

    if args.sample_with_offroad:
        output['offroad_prob'] = F.softmax(output['offroad_prob'], -1)
        off_ls = -torch.round(output['offroad_prob'][:, :, 0]) * output['speed'].view(-1, args.pred_step) + torch.round(output['offroad_prob'][:, :, 1]) * args.speed_threshold
        off_ls = (off_ls.view(-1, args.pred_step, 1) * weight).sum(-1).sum(-1)
        loss += off_ls

    return loss


def get_accuracy(output, target):
    tn, fp, fn, tp = confusion_matrix(output, target, labels=[0, 1]).ravel()
    score = (tn + tp) / (tn + fp + fn + tp) * 100.0
    return score

