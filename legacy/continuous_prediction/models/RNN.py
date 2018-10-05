import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weights_init

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        self.encoder = nn.Linear(args.hidden_dim + args.info_dim, args.hidden_dim)
        self.fc1 = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc4 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.lstm = nn.LSTMCell(args.hidden_dim, args.hidden_dim)
        self.out_encoder = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.apply(weights_init)

    def forward(self, feature_encoding, action, hx, cx):
        encoding = F.relu(self.encoder(torch.cat([feature_encoding, action], dim = 1)))
        x = F.relu(self.fc1(torch.cat([encoding, hx], dim = 1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        hx, cx = self.lstm(x, (hx, cx))
        hx = self.out_encoder(hx)
        return hx, cx