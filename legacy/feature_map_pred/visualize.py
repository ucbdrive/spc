import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import os
import drn
from single_conv import PRED as single_pred
from double_conv import PRED as double_pred
from separate_conv import PRED as separate_pred
from lstm_conv import PRED as lstm_pred
from dilate_and_conv import PRED as dilate_and_pred
from single_conv import UP
from seg import fill_up_weights, draw_from_pred
from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt

model_types = ['single', 'double', 'separate', 'dilate_and', 'lstm']
model_type = model_types[0]

def latest_model(model_type):
    model_list = os.listdir(model_type + '_conv_models')
    return sorted(model_list, key = lambda x: int(x[5:-4]))[-1]

def select(episode, step):
    model_list = os.listdir('dataset')
    data_name = list(filter(lambda x: ('episode%d_step%d' % (episode, step)) in x, model_list))[0]
    return os.path.join('dataset', data_name)

if __name__ == '__main__':
    model = eval(model_type + '_pred()')
    up_model = UP()
    model.load_state_dict(torch.load(os.path.join(model_type + '_conv_models', latest_model(model_type))))
    inputs = Variable(torch.from_numpy(np.load(select(0, 99))), requires_grad = False)
    if torch.cuda.is_available():
        model = model.cuda()
        up_model = up_model.cuda()
        inputs = inputs.cuda()
    preds = []

    action = 1
    output = inputs
    if model_type == 'lstm':
        hx = Variable(torch.zeros(1, 400))
        cx = Variable(torch.zeros(1, 400))
        if torch.cuda.is_available():
            hx = hx.cuda()
            cx = cx.cuda()
    for i in range(5):
        plt.subplot(4, 5, i + 1)
        if model_type == 'lstm':
            output, hx, cx = model(output, action, hx, cx)
        else:
            output = model(output, action)
        _, pred = torch.max(up_model(output), 1)
        plt.imshow(draw_from_pred(pred))
    for i in range(5):
        plt.subplot(4, 5, i + 11)
        if model_type == 'lstm':
            output, hx, cx = model(output, action, hx, cx)
        else:
            output = model(output, action)
        _, pred = torch.max(up_model(output), 1)
        plt.imshow(draw_from_pred(pred))

    for i in range(5):
        plt.subplot(4, 5, i + 6)
        inputs = Variable(torch.from_numpy(np.load(select(0, i + 100))), requires_grad = False)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        _, pred = torch.max(up_model(inputs), 1)
        plt.imshow(draw_from_pred(pred))
    for i in range(5):
        plt.subplot(4, 5, i + 16)
        inputs = Variable(torch.from_numpy(np.load(select(0, i + 105))), requires_grad = False)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        _, pred = torch.max(up_model(inputs), 1)
        plt.imshow(draw_from_pred(pred))

    '''
    for action in range(6):
        plt.subplot(2, 3, action + 1)
        if model_type == 'lstm':
            hx = Variable(torch.zeros(1, 400))
            cx = Variable(torch.zeros(1, 400))
            if torch.cuda.is_available():
                hx = hx.cuda()
                cx = cx.cuda()
            output, hx, cx = model(inputs, action, hx, cx)
            for i in range(10):
                output, hx, cx = model(output, action, hx, cx)
        else:
            output = model(inputs, action)
            for i in range(10):
                output = model(output, action)
            output = inputs
        _, pred = torch.max(up_model(output), 1)
        preds.append(pred)
        plt.imshow(draw_from_pred(pred))
    '''
    plt.savefig('pred_%s.png' % model_type, dpi = 300)
