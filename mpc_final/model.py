''' Model Definition '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import drn
import dla
import math

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias = self.bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias = self.bias)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias = self.bias)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim, bias = self.bias)
        self.W = nn.Linear(hidden_dim, 4 * hidden_dim, bias = self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim = 1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        combined_conv = F.relu(self.W(x))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim = 1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next # h_next is the output

    def init_hidden(self, batch_size):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            return (Variable(torch.zeros(batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.hidden_dim)),
                    Variable(torch.zeros(batch_size, self.hidden_dim)))

class DRNSeg(nn.Module):
    ''' Network For Feature Extraction for Segmentation Prediction '''
    def __init__(self, model_name, num_classes, pretrained=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.seg = nn.Conv2d(model.out_dim, num_classes, kernel_size=1, bias=True)
        
        # initialization
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        self.seg.weight.data.normal_(0, math.sqrt(2. / n))
        self.seg.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        feature_map = self.seg(x)
        return feature_map # size: None * 4 * 32 * 32

class UP_Sampler(nn.Module):
    def __init__(self, num_classes):
        super(UP_Sampler, self).__init__()
        self.up = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4,
                                output_padding=0, groups=4, bias=False)
        
    def forward(self, feature_map):
        out = self.up(feature_map)
        return out # segmentation

class ConvLSTMNet(nn.Module):
    def __init__(self, num_actions=3,
                pretrained=True,
                frame_history_len=4,
                use_seg=True,
                use_xyz=True,
                model_name=None,
                num_classes=4,
                hidden_dim=512,
                info_dim=16):
        super(ConvLSTMNet, self).__init__()
        self.frame_history_len = frame_history_len
        self.use_seg = use_seg
        self.use_xyz = use_xyz
        self.hidden_dim = hidden_dim
        self.info_dim = info_dim

        # feature extraction part
        if use_seg:
            self.drnseg = DRNSeg(model_name, num_classes, pretrained)
            self.up_sampler = UP_Sampler(num_classes)
            self.feature_encode = nn.Linear(4096 * frame_history_len, self.hidden_dim)
            self.up_scale = nn.Linear(self.hidden_dim, 4096 * frame_history_len)
        else:
            self.dla = dla.dla46x_c(pretrained=pretrained)
            self.feature_encode = nn.Linear(256 * frame_history_len, self.hidden_dim)
        self.outfeature_encode = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lstm = ConvLSTMCell(self.hidden_dim, self.hidden_dim, True)
        self.action_encode = nn.Linear(num_actions, info_dim)
        self.info_encode = nn.Linear(self.info_dim+self.hidden_dim, self.hidden_dim)
        
        # output layer
        self.fc_coll_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_coll_2 = nn.Linear(128+info_dim, 32)
        self.fc_coll_3 = nn.Linear(32+info_dim, 2)
        self.fc_off_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_off_2 = nn.Linear(128+info_dim, 32)
        self.fc_off_3 = nn.Linear(32+info_dim, 2)
        self.fc_dist_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_dist_2 = nn.Linear(128+info_dim, 32)
        self.fc_dist_3 = nn.Linear(32+info_dim, 1)
        if self.use_xyz:
            self.fc_xyz_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
            self.fc_xyz_2 = nn.Linear(128+info_dim, 32)
            self.fc_xyz_3 = nn.Linear(32+info_dim, 3)   
     
    def get_feature(self, x):
        res = []
        batch_size = x.size(0)
        if self.use_seg:
            for i in range(self.frame_history_len):
                out = self.drnseg(x[:, i*3 : (i+1)*3, :, :])
                out = out.squeeze().view(batch_size, -1)
                res.append(out)
        else:
            for i in range(self.frame_history_len):
                out = self.dla(x[:, i*3 : (i+1)*3, :, :])
                out = out.squeeze().view(batch_size, -1)
                res.append(out)
        res = torch.cat(res, dim=1)
        res = self.feature_encode(res)
        return res 
    
    def pred_seg(self, x):
        res = []
        batch_size = x.size(0)
        x = x.view(batch_size, self.frame_history_len*4, 32, 32)
        for i in range(self.frame_history_len):
            out = self.up_sampler(x[:, i*3:(i+1)*3, :, :])
            res.append(out)
        res = torch.cat(res, dim=1)
        return res
     
    def forward(self, x, action, with_encode=False, hidden=None, cell=None):
        if with_encode == False:
            x = self.get_feature(x)
        if hidden is None or cell is None:
            hidden, cell = x, x
        info_enc = F.relu(self.action_encode(action))
        action_enc = info_enc
        encode = torch.cat([x, info_enc], dim=1)
        encode = F.relu(self.info_encode(encode))
        hidden, cell = self.lstm(encode, [hidden, cell])
        
        pred_encode_nx = hidden.view(-1, self.hidden_dim)
        nx_feature_enc = self.outfeature_encode(F.relu(pred_encode_nx))
        hidden_enc = torch.cat([pred_encode_nx, info_enc], dim=1)
        
        if self.use_seg:
            seg_feat = self.up_scale(F.relu(nx_feature_enc))
            seg_pred = self.pred_seg(seg_feat)
        else:
            seg_pred = nx_feature_enc

        coll_prob = F.relu(self.fc_coll_1(hidden_enc))
        coll_prob = torch.cat([coll_prob, action_enc], dim = 1)
        coll_prob = F.relu(self.fc_coll_2(coll_prob))
        coll_prob = torch.cat([coll_prob, action_enc], dim = 1)
        coll_prob = nn.Softmax(dim=-1)(F.relu(self.fc_coll_3(coll_prob)))

        offroad_prob = F.relu(self.fc_off_1(hidden_enc))
        offroad_prob = torch.cat([offroad_prob, action_enc], dim=1)
        offroad_prob = F.relu(self.fc_off_2(offroad_prob))
        offroad_prob = torch.cat([offroad_prob, action_enc], dim=1)
        offroad_prob = nn.Softmax(dim=-1)(F.relu(self.fc_off_3(offroad_prob)))

        dist = F.relu(self.fc_dist_1(hidden_enc))
        dist = torch.cat([dist, action_enc], dim=1)
        dist = F.relu(self.fc_dist_2(dist))
        dist = torch.cat([dist, action_enc], dim=1)
        dist = self.fc_dist_3(dist)

        if self.use_xyz:
            xyz = F.relu(self.fc_xyz_1(hidden_enc))
            xyz = torch.cat([xyz, action_enc], dim=1)
            xyz = F.relu(self.fc_xyz_2(xyz))
            xyz = torch.cat([xyz, action_enc], dim=1)
            xyz = self.fc_xyz_3(xyz)
        else:
            xyz = None
        
        return coll_prob, seg_pred, offroad_prob, dist, xyz, hidden, cell 
