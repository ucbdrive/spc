from utils import *
from model import *
from torch.autograd import Variable
import torch
import pdb

class dummy():
    def __init__(self):
        self.pretrained = None
        self.drn_model = 'dla46x_c'
        self.classes = 4
        self.frame_history_len = 3
        self.pred_step = 10
        self.num_total_act = 2
        self.one_hot = True
        self.use_collision = True
        self.use_offroad = True
        self.use_distance = True
        self.use_seg = True
        self.lstm2 = True
        self.hidden_dim = 1024
        self.info_dim = 32
        self.use_lstm = False
        self.use_otherlane = False
        self.use_pos = False
        self.use_angle = False
        self.use_speed = False
        self.use_xyz = False
        self.use_dqn = False
        self.num_dqn_action = 10
        
segnet = ConvLSTMMulti(dummy())
segnet.eval()
state_dict = torch.load('trained_model_torcs.pth')
segnet.load_state_dict(state_dict)
segnet = torch.nn.DataParallel(segnet).cuda().float()
obs = Variable(torch.zeros((3, 256, 256))).unsqueeze(0).cuda().float()
output = segnet(obs)
pdb.set_trace()

