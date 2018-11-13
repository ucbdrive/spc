import os
import argparse
from args import init_parser
import torch
from model import ConvLSTMMulti

parser = argparse.ArgumentParser(description='Train-torcs')
init_parser(parser)  # See `args.py` for default arguments
args = parser.parse_args()
args.continuous = True
args.use_seg = True
args.use_collision = True
args.use_offroad = True
args.use_distance = True
args.lstm2 = True
args.one_hot = True

def init_model(args):
    net = ConvLSTMMulti(args)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    state_dict = torch.load('trained_model.pth')
    net.load_state_dict(state_dict)

    if torch.cuda.is_available():
        net = net.cuda()

    return net

if __name__ == '__main__':
    net = init_model(args)
    dla = net.conv_lstm.drnseg
    torch.save(dla.state_dict(), 'dlaseg.pth')
