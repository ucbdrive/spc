''' Model Definition '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import drn
import dla
import math
import pdb
from PRED import PRED

class atari_model(nn.Module):
    def __init__(self, inc=12, num_actions=9, frame_history_len=4):
        super(atari_model, self).__init__()
        self.conv1 = nn.Conv2d(inc, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.frame_history_len = frame_history_len
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        res = self.fc5(x)
        return res

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
    def __init__(self, model_name, num_classes=4, pretrained=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(pretrained=pretrained, num_classes=1000)
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
    def __init__(self, num_actions=2,
                pretrained=True,
                frame_history_len=4,
                use_seg=True,
                use_xyz=True,
                model_name='drn_d_22',
                num_classes=4,
                hidden_dim=512,
                info_dim=16):
        super(ConvLSTMNet, self).__init__()
        self.frame_history_len = frame_history_len
        self.use_seg = use_seg
        self.use_xyz = use_xyz
        self.hidden_dim = hidden_dim
        self.info_dim = info_dim
        self.num_classes = num_classes

        # feature extraction part
        if use_seg:
            self.drnseg = DRNSeg(model_name, num_classes, pretrained)
            self.up_sampler = UP_Sampler(num_classes)
            self.feature_map_conv1 = nn.Conv2d(num_classes * frame_history_len, 16, 5, stride = 1, padding = 2)
            self.feature_map_conv2 = nn.Conv2d(16, 32, 3, stride = 1, padding = 1)
            self.feature_map_conv3 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
            self.feature_map_fc1 = nn.Linear(1024, 1024)
            self.feature_map_fc2 = nn.Linear(1024, self.hidden_dim)
            self.up_scale = nn.Linear(self.hidden_dim+self.info_dim, 32*32*num_classes)
            self.pred = PRED(num_classes, num_actions)
        else:
            self.dla = dla.dla46x_c(pretrained=pretrained)
            self.feature_encode = nn.Linear(256 * frame_history_len, self.hidden_dim)
            self.outfeature_encode = nn.Linear(self.hidden_dim+self.info_dim, self.hidden_dim)
        self.lstm = ConvLSTMCell(self.hidden_dim, self.hidden_dim, True)
        self.action_encode = nn.Linear(num_actions, info_dim)
        self.info_encode = nn.Linear(self.info_dim + self.hidden_dim, self.hidden_dim)
        
        # output layer
        self.fc_coll_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_coll_2 = nn.Linear(128 + info_dim, 32)
        self.fc_coll_3 = nn.Linear(32 + info_dim, 2)
        self.fc_off_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_off_2 = nn.Linear(128 + info_dim, 32)
        self.fc_off_3 = nn.Linear(32 + info_dim, 2)
        self.fc_dist_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_dist_2 = nn.Linear(128 + info_dim, 32)
        self.fc_dist_3 = nn.Linear(32 + info_dim, 1)
        if self.use_xyz:
            self.fc_xyz_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
            self.fc_xyz_2 = nn.Linear(128 + info_dim, 32)
            self.fc_xyz_3 = nn.Linear(32 + info_dim, 3)
     
    def pred_seg(self, x, action):
        res = []
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_classes, 32, 32)
        x = self.pred(x, action)
        out = self.up_sampler(x)
        return out

    def forward(self, x, action, with_encode=False, hidden=None, cell=None):
        if with_encode == False:
            x = self.get_feature(x)
            if self.use_seg:
                x = F.tanh(F.max_pool2d(self.feature_map_conv1(x), kernel_size = 2, stride = 2))
                x = F.tanh(F.max_pool2d(self.feature_map_conv2(x), kernel_size = 2, stride = 2))
                x = F.tanh(F.max_pool2d(self.feature_map_conv3(x), kernel_size = 2, stride = 2)) # 1x64x4x4
                x = x.view(x.size(0), -1) # 1024
                x = F.relu(self.feature_map_fc1(x))
                x = F.relu(self.feature_map_fc2(x))
        if hidden is None or cell is None:
            hidden, cell = x, x
        action_enc = F.relu(self.action_encode(action))
        encode = torch.cat([x, action_enc], dim = 1)
        encode = F.relu(self.info_encode(encode))
        hidden, cell = self.lstm(encode, [hidden, cell])
    
        # this is to be output as feature representation for next step no matter with seg or not
        nx_feature_enc = hidden.view(-1, self.hidden_dim)
        hidden_enc = torch.cat([nx_feature_enc, action_enc], dim = 1)
       
        ''' next feature encoding: seg_pred ''' 
        if self.use_seg:
            seg_feat = self.up_scale(F.relu(hidden_enc))
            seg_pred = self.pred_seg(seg_feat, action)
        else:
            seg_pred = self.outfeature_encode(F.relu(hidden_enc))

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
        
        return coll_prob, nx_feature_enc, offroad_prob, dist, xyz, seg_pred, hidden, cell
     
    def get_feature(self, x):
        res = []
        batch_size = x.size(0)
        if self.use_seg:
            res = torch.cat([self.drnseg(x[:, i*3 : (i+1)*3, :, :]) for i in range(self.frame_history_len)], dim = 1)
        else:
            for i in range(self.frame_history_len):
                out = self.dla(x[:, i*3 : (i+1)*3, :, :])
                out = out.squeeze().view(batch_size, -1)
                res.append(out)
            res = torch.cat(res, dim = 1)
            res = self.feature_encode(res)
        return res 

class ConvLSTMMulti(nn.Module):
    def __init__(self, num_actions = 2,
                pretrained = True,
                frame_history_len = 4,
                use_seg = True,
                use_xyz = True,
                model_name = 'drn_d_22',
                num_classes = 4,
                hidden_dim = 512, 
                info_dim = 16):
        super(ConvLSTMMulti, self).__init__()
        self.conv_lstm = ConvLSTMNet(num_actions, pretrained, frame_history_len,
                                    use_seg, use_xyz, model_name, num_classes, hidden_dim, info_dim)
        self.frame_history_len = frame_history_len
        self.num_actions = num_actions

    def get_feature(self, x):
        feat = []
        x = x.contiguous()
        num_time = int(x.size()[1])
        for i in range(num_time):
            feat.append(self.conv_lstm.get_feature(x[:, i, :, :, :].squeeze(1)))
        return torch.stack(feat, dim = 1)

    def forward(self, imgs, actions=None, num_time=None, hidden=None, cell=None, get_feature=False):
        if get_feature:
            res = self.get_feature(imgs)
            return res
        batch_size, num_step, c, w, h = int(imgs.size()[0]), int(imgs.size()[1]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
        coll, pred, offroad, dist, xyz, seg_pred, hidden, cell = self.conv_lstm(imgs[:,0,:,:,:].squeeze(1), actions[:,0,:].squeeze(1), hidden=hidden, cell=cell)
        num_act = self.num_actions
        coll_list = [coll]
        pred_list = [pred]
        offroad_list = [offroad]
        dist_list = [dist]
        xyz_list = [xyz]
        seg_list = [seg_pred]
        for i in range(1, num_time):
            coll, pred, offroad, dist, xyz, seg_pred, hidden, cell = self.conv_lstm(pred, actions[:,i,:].squeeze(1), with_encode=True, hidden=hidden, cell=cell)
            coll_list.append(coll)
            pred_list.append(pred)
            offroad_list.append(offroad)
            dist_list.append(dist)
            xyz_list.append(xyz)
            seg_list.append(seg_pred)
        seg_out = torch.stack(seg_list, dim=1)
        if self.conv_lstm.use_xyz:
            xyz_out = torch.stack(xyz_list, dim=1)
        else:
            xyz_out = None
        return torch.stack(coll_list, dim=1), torch.stack(pred_list, dim=1), torch.stack(offroad_list,dim=1), torch.stack(dist_list, dim=1), xyz_out, seg_out, hidden, cell

if __name__ == '__main__':
    net = ConvLSTMMulti(num_actions=2, pretrained=True, frame_history_len=4, use_seg=True,
                      use_xyz=True, model_name='drn_d_22', num_classes=4, hidden_dim=512, info_dim=16)
    action = Variable(torch.zeros(16, 2, 2), requires_grad = False)
    img = Variable(torch.zeros(16, 2, 12, 256, 256), requires_grad = False)
    result = net(img, action)
    for i in result:
        print(i.size())


