import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import random
import numpy as np
import os
import drn2
from seg import fill_up_weights, draw_from_pred
from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt
from py_TORCS import torcs_envs

def naive_driver(info):
    if info['angle'] > 0.5 or (info['trackPos'] < -1 and info['angle'] > 0):
        return 0
    elif info['angle'] < -0.5 or (info['trackPos'] > 3 and info['angle'] < 0):
        return 2
    return 1

def Focal_Loss(probs, target, reduce=True):
    # probs : batch * num_class
    # target : batch,
    loss = -1.0 * (1-probs).pow(1) * torch.log(probs)
    batch_size = int(probs.size()[0])
    loss = loss[torch.arange(batch_size).long().cuda(), target.long()]
    if reduce == True:
        loss = loss.sum()/(batch_size*1.0)
    return loss

def Focal_Loss_Regress(probs, target, reduce=True):
    # probs: batch * num_class
    # target: batch * num_class
    target_class = (torch.max(target, -1)[1]).view(-1,1)
    target = target * 2 - 1
    res1 = -1.0*(torch.log(probs) * target)
    weight = Variable(torch.arange(19)).view(1,19).repeat(probs.size()[0], 1).cuda().float()
    weight = 0.1*torch.abs(weight - target_class.repeat(1, 19).float())+1
    loss = (weight * res1).sum(-1)
    if reduce == True:
        loss = loss.sum()/(probs.size()[0]*19.0)
    return loss

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=False, use_torch_up=False, num_actions = 6):
        super(DRNSeg, self).__init__()
        model = drn2.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        self.up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                output_padding=0, groups=classes, bias=False)
        fill_up_weights(self.up)

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return F.log_softmax(y, dim = 1), x # , dim = 1

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
        for param in self.up.parameters():
            yield param

class PRED(nn.Module):
    def __init__(self):
        super(PRED, self).__init__()
        self.action_encoder1 = nn.Linear(6, 16*4*5*5)
        self.action_encoder1.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.action_encoder2 = nn.Linear(6, 16*16*5*5)
        self.action_encoder2.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()

        self.bn2 = nn.BatchNorm2d(16)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()

        self.action_encoder3 = nn.Linear(6, 32*16*5*5)
        self.action_encoder3.weight.data.normal_(0, math.sqrt(2. / (32*5*5)))

        self.action_encoder4 = nn.Linear(6, 32*32*5*5)
        self.action_encoder4.weight.data.normal_(0, math.sqrt(2. / (32*5*5)))

        self.bn3 = nn.BatchNorm2d(32)
        self.bn3.weight.data.fill_(1)
        self.bn3.bias.data.zero_()

        self.bn4 = nn.BatchNorm2d(32)
        self.bn4.weight.data.fill_(1)
        self.bn4.bias.data.zero_()

        self.action_encoder5 = nn.Linear(6, 64*32*5*5)
        self.action_encoder5.weight.data.normal_(0, math.sqrt(2. / (64*5*5)))

        self.action_encoder6 = nn.Linear(6, 64*64*5*5)
        self.action_encoder6.weight.data.normal_(0, math.sqrt(2. / (64*5*5)))

        self.bn5 = nn.BatchNorm2d(64)
        self.bn5.weight.data.fill_(1)
        self.bn5.bias.data.zero_()

        self.bn6 = nn.BatchNorm2d(64)
        self.bn6.weight.data.fill_(1)
        self.bn6.bias.data.zero_()

        self.action_encoder7 = nn.Linear(6, 4*64*1*1)
        self.action_encoder7.weight.data.normal_(0, math.sqrt(2. / (4*1*1)))

    def forward(self, x, action):
        one_hot = torch.zeros(1, 6)
        one_hot[0, action] = 1
        one_hot = Variable(one_hot, requires_grad = False)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        y1 = self.action_encoder1(one_hot).view(16, 4, 5, 5)
        y2 = self.action_encoder2(one_hot).view(16, 16, 5, 5)
        y3 = self.action_encoder3(one_hot).view(32, 16, 5, 5)
        y4 = self.action_encoder4(one_hot).view(32, 32, 5, 5)
        y5 = self.action_encoder5(one_hot).view(64, 32, 5, 5)
        y6 = self.action_encoder6(one_hot).view(64, 64, 5, 5)
        y7 = self.action_encoder7(one_hot).view(4, 64, 1, 1)

        result = F.relu(self.bn1(F.conv2d(x, y1, stride = 1, padding = 4, dilation = 2)))
        result = self.bn2(F.conv2d(result, y2, stride = 1, padding = 2))
        result = F.relu(self.bn3(F.conv2d(result, y3, stride = 1, padding = 4, dilation = 2)))
        result = self.bn4(F.conv2d(result, y4, stride = 1, padding = 2))
        result = F.relu(self.bn5(F.conv2d(result, y5, stride = 1, padding = 4, dilation = 2)))
        result = self.bn6(F.conv2d(result, y6, stride = 1, padding = 2))
        result = F.conv2d(result, y7, stride = 1, padding = 0)
        return result

class FURTHER(nn.Module):
    def __init__(self):
        super(FURTHER, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 5, stride = 1, padding = 2)
        self.conv1.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.conv2 = nn.Conv2d(16, 32, 3, stride = 1, padding = 1)
        self.conv2.weight.data.normal_(0, math.sqrt(2. / (32*3*3)))

        self.conv3 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.conv3.weight.data.normal_(0, math.sqrt(2. / (64*3*3)))

        self.fc1 = nn.Linear(4480, 1024)
        self.fc1.weight.data.normal_(0, math.sqrt(2. / 1024))

        self.fc2 = nn.Linear(1024, 256)
        self.fc2.weight.data.normal_(0, math.sqrt(2. / 256))

        self.fc3_1 = nn.Linear(256, 64)
        self.fc3_1.weight.data.normal_(0, math.sqrt(2. / 64))

        self.fc3_2 = nn.Linear(256, 64)
        self.fc3_2.weight.data.normal_(0, math.sqrt(2. / 64))

        self.fc3_3 = nn.Linear(256, 64)
        self.fc3_3.weight.data.normal_(0, math.sqrt(2. / 64))

        self.fc4_1 = nn.Linear(64, 16)
        self.fc4_1.weight.data.normal_(0, math.sqrt(2. / 16))

        self.fc4_2 = nn.Linear(64, 16)
        self.fc4_2.weight.data.normal_(0, math.sqrt(2. / 16))

        self.fc4_3 = nn.Linear(64, 16)
        self.fc4_3.weight.data.normal_(0, math.sqrt(2. / 16))

        self.fc5_1 = nn.Linear(16, 1)
        self.fc5_1.weight.data.normal_(0, math.sqrt(2. / 1))

        self.fc5_2 = nn.Linear(16, 1)
        self.fc5_2.weight.data.normal_(0, math.sqrt(2. / 1))

        self.fc5_3 = nn.Linear(16, 1)
        self.fc5_3.weight.data.normal_(0, math.sqrt(2. / 1))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size = 2, stride = 2)) # 1x64x7x10
        x = x.view(x.size(0), -1) # 4480
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pos = self.fc5_1(F.relu(self.fc4_1(F.relu(self.fc3_1(x))))).view(1)
        angle = self.fc5_2(F.relu(self.fc4_2(F.relu(self.fc3_2(x))))).view(1)
        speed = self.fc5_3(F.relu(self.fc4_3(F.relu(self.fc3_3(x))))).view(1)
        return pos, angle, speed

# def latest_model(dir = 'pred_models'):
#     model_list = os.listdir(dir)
#     return os.path.join(dir, sorted(model_list, key = lambda x: int(x[5:-4]))[-1])

seed = 233
num_steps = 12
obs_avg = 112.62289744791671
obs_std = 56.1524832523
train = True
LOAD = False
latest_epoch = 140

if __name__ == '__main__':
    random.seed(seed)
    torch.manual_seed(seed)
    if os.path.exists('pred_log.txt'):
        os.system('rm pred_log.txt')
    if not os.path.isdir('pred_models'):
        os.mkdir('pred_models')
    else:
        os.system('rm pred_models/*.dat')
    model = DRNSeg('drn_d_22', 4)
    predictor = PRED()
    further = FURTHER()

    env = torcs_envs(num = 1, isServer = 0, mkey_start = 817).get_envs()[0]
    obs = env.reset()
    obs, reward, done, info = env.step(1)
    all_obs = np.repeat(np.expand_dims(obs, axis = 0), num_steps + 1, axis = 0)
    obs = (obs.transpose(2, 0, 1) - obs_avg) / obs_std

    true_obs = np.repeat(obs, 3, axis = 0)
    obs_list = np.repeat(np.expand_dims(true_obs, axis = 0), num_steps + 1, axis = 0)
    seg = env.get_segmentation().astype(np.uint8)
    seg_list = np.repeat(np.expand_dims(seg, axis = 0), num_steps + 1, axis = 0)
    action_array = np.repeat(np.array([4]), num_steps)
    pos_array = np.ones(num_steps + 1) * info['trackPos'] / 7.0
    angle_array = np.ones(num_steps + 1) * info['angle'] * 2.0
    speed_array = np.ones(num_steps + 1) * info['speed'] / 20.0

    if not train or LOAD:
        model.load_state_dict(torch.load(os.path.join('pred_models', 'model_epoch%d.dat' % latest_epoch)))
        predictor.load_state_dict(torch.load(os.path.join('pred_models', 'predictor_epoch%d.dat' % latest_epoch)))
        further.load_state_dict(torch.load(os.path.join('pred_models', 'further_epoch%d.dat' % latest_epoch)))
    if train:
        inputs = torch.autograd.Variable(torch.ones(num_steps + 1, 9, 480, 640), requires_grad = False)
        output = torch.autograd.Variable(torch.ones(num_steps + 1, 4, 480, 640), requires_grad = False)
        target = torch.autograd.Variable(torch.ones(num_steps + 1, 480, 640), requires_grad = False).type(torch.LongTensor)
        target_pos = torch.autograd.Variable(torch.zeros(num_steps + 1), requires_grad = False)
        target_angle = torch.autograd.Variable(torch.zeros(num_steps + 1), requires_grad = False)
        target_speed = torch.autograd.Variable(torch.zeros(num_steps + 1), requires_grad = False)
        NLL = nn.NLLLoss2d()
        BCE = nn.BCELoss()
        L1 = nn.SmoothL1Loss()
        L2 = nn.MSELoss()
        optimizer = torch.optim.Adam(list(model.optim_parameters()) + list(predictor.parameters()) + list(further.parameters()),
                                    0.0001)
    else:
        inputs = torch.autograd.Variable(torch.ones(1, 9, 480, 640), requires_grad = False)

    if torch.cuda.is_available():
        model = model.cuda()
        predictor = predictor.cuda()
        further = further.cuda()
        inputs = inputs.cuda()
        output = output.cuda()
        if train:
            target = target.cuda()
            target_pos = target_pos.cuda()
            target_angle = target_angle.cuda()
            target_speed = target_speed.cuda()
            NLL = NLL.cuda()
            BCE = BCE.cuda()
            L1 = L1.cuda()
            L2 = L2.cuda()
    
    if train:
        model.train()
        predictor.train()
        further.train()
        losses = 0
        epoch = 0
        while True: # while
            LOSS = np.zeros((num_steps + 1, 5))

            for i in range(600): # 2000
                action = random.randint(0, 5) if random.random() < 0.5 else naive_driver(info)
                obs, reward, done, info = env.step(action)
                all_obs = np.concatenate((all_obs[1:], np.expand_dims(obs, axis = 0)), axis = 0)
                obs = (obs.transpose(2, 0, 1) - obs_avg) / obs_std
                true_obs = np.concatenate((true_obs[3:], obs), axis=0)
                obs_list = np.concatenate((obs_list[1:], true_obs[np.newaxis]), axis = 0)

                seg = np.expand_dims(env.get_segmentation().astype(np.uint8), axis = 0)
                seg_list = np.concatenate((seg_list[1:], seg), axis = 0)

                action_array = np.concatenate((action_array[1:], np.array([action])))

                pos_array = np.concatenate((pos_array[1:], np.array([info['trackPos'] / 7.0])))
                angle_array = np.concatenate((angle_array[1:], np.array([info['angle'] * 2.0])))
                speed_array = np.concatenate((speed_array[1:], np.array([info['speed'] / 20.0])))

                inputs[:] = torch.from_numpy(obs_list)
                target[:] = torch.from_numpy(seg_list).type(torch.LongTensor)
                target_pos[:] = torch.from_numpy(pos_array)
                target_angle[:] = torch.from_numpy(angle_array)
                target_speed[:] = torch.from_numpy(speed_array)
                output[0], feature_map = model(inputs[0].view(1, 9, 480, 640))
                pos, angle, speed = further(feature_map)
                loss0 = NLL(output[0].view(1, 4, 480, 640), target[0].view(1, 480, 640))
                loss1 = 0
                loss2 = L2(pos, target_pos[0])
                loss3 = L2(angle, target_angle[0])
                loss4 = L2(speed, target_speed[0])
                loss = loss2 + loss3 + loss4
                LOSS[0, 0] += loss0.data.cpu().numpy()
                # LOSS[0, 1] += loss1.data.cpu().numpy()[0]
                LOSS[0, 2] += loss2.data.cpu().numpy()
                LOSS[0, 3] += loss3.data.cpu().numpy()
                LOSS[0, 4] += loss4.data.cpu().numpy()
                gamma = 1
                for j in range(1, num_steps + 1):
                    gamma *= 0.97
                    _, target_feature_map = model(inputs[j].view(1, 9, 480, 640))
                    target_feature_map = target_feature_map.detach()
                    target_feature_map.requires_grad = False
                    target_feature_map = target_feature_map.cuda()
                    feature_map = predictor(feature_map, action_array[j - 1])
                    pos, angle, speed = further(feature_map)
                    output[j] = F.log_softmax(model.up(feature_map), dim = 1) # dim = 2
                    loss0 = NLL(output[j].view(1, 4, 480, 640), target[j].view(1, 480, 640))
                    loss1 = L2(feature_map, target_feature_map) / 1000
                    loss2 = L2(pos, target_pos[j])
                    loss3 = L2(angle, target_angle[j])
                    loss4 = L2(speed, target_speed[j])
                    loss += gamma * (loss1 + loss2 + loss3 + loss4)
                    LOSS[j, 0] += loss0.data.cpu().numpy()
                    LOSS[j, 1] += loss1.data.cpu().numpy()
                    LOSS[j, 2] += loss2.data.cpu().numpy()
                    LOSS[j, 3] += loss3.data.cpu().numpy()
                    LOSS[j, 4] += loss4.data.cpu().numpy()
                losses += loss.data.cpu().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if done or reward <= -2.5:
                    obs = env.reset()
                    obs, reward, done, info = env.step(1)
                    all_obs = np.repeat(np.expand_dims(obs, axis = 0), num_steps + 1, axis = 0)
                    true_obs = np.repeat((obs.transpose(2, 0, 1) - obs_avg) / obs_std, 3, axis = 0)

                    obs_list = np.repeat(np.expand_dims(true_obs, axis = 0), num_steps + 1, axis = 0)
                    seg = env.get_segmentation().astype(np.uint8)
                    seg_list = np.repeat(np.expand_dims(seg, axis = 0), num_steps + 1, axis = 0)
                    action_array = np.repeat(np.array([4]), num_steps)
                    pos_array = np.ones(num_steps + 1) * info['trackPos'] / 7.0
                    angle_array = np.ones(num_steps + 1) * info['angle'] * 2.0
                    speed_array = np.ones(num_steps + 1) * info['speed'] / 20.0

                if (i + 1) % 20 == 0:
                    pred = torch.argmax(output, dim = 1)
                    print('Saving images.')
                    for i in range(num_steps + 1):
                        imsave('pred_images/%d.png' % i, np.concatenate((all_obs[i], draw_from_pred(target[i]), draw_from_pred(pred[i])), axis = 1))
                if (i + 1) % 100 == 0:
                    print(LOSS / 100.0)
                    print('iteration %d, mean loss %f' % (i + 1, losses / 100.0))
                    with open('pred_log.txt', 'a') as f:
                        f.write('%s\nIteration %d, mean loss %f\n' % (str(LOSS / 100.0), i + 1, losses / 100.0))
                    losses = 0
                    LOSS = np.zeros((num_steps + 1, 5))
                if (i + 1) % 200 == 0:
                    torch.save(model.state_dict(), os.path.join('pred_models', 'model_epoch%d.dat' % epoch))
                    torch.save(predictor.state_dict(), os.path.join('pred_models', 'predictor_epoch%d.dat' % epoch))
                    torch.save(further.state_dict(), os.path.join('pred_models', 'further_epoch%d.dat' % epoch))
            epoch += 1
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        for param in predictor.parameters():
            param.requires_grad = False
        predictor.eval()
        for param in further.parameters():
            param.requires_grad = False
        further.eval()

        env = torcs_envs(num = 1, isServer = 0).get_envs()[0]
        obs = env.reset()
        obs, reward, done, info = env.step(1)
        true_obs = np.repeat((obs.transpose(2, 0, 1) - obs_avg) / obs_std, 3, axis = 0)
        for i in range(300):
            inputs[:] = torch.from_numpy(true_obs)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output, feature_map = model(inputs)
            pos, angle, speed = further(feature_map)
            print(pos.data.cpu().numpy() * 7, angle.data.cpu().numpy(), speed.data.cpu().numpy())
            print(info['trackPos'], info['angle'], info['speed'])
            print()
            action = naive_driver({'trackPos': pos.data.cpu().numpy() * 7, 'angle':angle.data.cpu().numpy()})

            obs, reward, done, info = env.step(action)
            obs = (obs.transpose(2, 0, 1) - obs_avg) / obs_std
            true_obs = np.concatenate((true_obs[3:], obs), axis = 0)
        env.close()
