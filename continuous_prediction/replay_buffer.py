import numpy as np
import torch
from utils import init_training_variables

class replay_buffer(object):
    def __init__(self, args):
        super(replay_buffer, self).__init__()
        self.args = args
        self.data_dict = dict()
        self.index, self.real_size = 0, 0
        self.data_dict['done'] = np.zeros(args.buffer_size)
        self.data_dict['obs'] = np.zeros((args.buffer_size, 3 * args.frame_len, 256, 256))

        if args.continuous:
            self.data_dict['action'] = np.zeros((args.buffer_size, args.action_dim))
        else:
            self.data_dict['action'] = np.zeros(args.buffer_size).astype(np.uint8)

        if args.use_dqn:
            self.data_dict['dqn_action'] = np.zeros((args.buffer_size, 1)).astype(np.uint8)
            self.data_dict['reward'] = np.zeros(args.buffer_size)

        if args.use_seg:
            self.data_dict['target_seg'] = np.zeros((args.buffer_size, 256, 256)).astype(np.uint8)

        self.data_dict['target_coll'] = np.zeros(args.buffer_size)

        self.data_dict['target_off'] = np.zeros(args.buffer_size)

        self.data_dict['target_dist'] = np.zeros(args.buffer_size)

        if args.use_xyz:
            self.data_dict['target_xyz'] = np.zeros((args.buffer_size, 3))

    def store(self, data):
        for key in data.keys():
            self.data_dict[key][self.index] = data[key]

        self.index = (self.index + 1) % self.args.buffer_size
        self.real_size += 1

    def sample_one_index(self):
        index = np.random.randint(min(self.args.buffer_size, self.real_size) - self.args.num_steps + 1)
        while np.sum(self.data_dict['done'][index: index + self.args.num_steps]) > 0:
            index = np.random.randint(min(self.args.buffer_size, self.real_size) - self.args.num_steps + 1)
        return index

    def sample(self):
        train_data = init_training_variables(self.args)
        indices = [self.sample_one_index() for i in range(self.args.batch_size)]
        for key in self.data_dict.keys():
            if key != 'done':
                for i in range(self.args.batch_size):
                    train_data[key][:, i] = torch.from_numpy(self.data_dict[key][indices[i]: indices[i] + self.args.num_steps])
        return train_data