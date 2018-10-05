from model import *
from envs import *
import argparse
from dqn_utils import *
import os
from torch.autograd import Variable
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--env-id', type=str, default='EnduroNoFrameskip-v0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--wrap-deepmind', action='store_true')
parser.add_argument('--frame-history-len', type=int, default=4)
parser.add_argument('--change-model', action='store_true')
parser.add_argument('--save-path', type=str, default='')
parser.add_argument('--device', type=str, default='cuda')

if __name__ == '__main__':
    args = parser.parse_args()
    set_global_seeds(args.seed)
    env = create_atari_env(args.env_id, args.seed, deepmind_wrap=args.wrap_deepmind)

    model = atari_model(env.observation_space.shape[-1]*args.frame_history_len, 
                        env.action_space.n, change=args.change_model)
    model.to(args.device).float()
    
    # load model
    model_path = args.save_path+'/dqn/model'
    file_list = sorted(os.listdir(model_path))
    file_name = os.path.join(model_path, 'model_0.pt')
    model.load_state_dict(torch.load(os.path.join(file_name)), map_location=args.device)
    
    # test
    obs = env.reset()
    rbuffer = ReplayBuffer(1000, args.frame_history_len)
    done = False
    r = 0
    while done == False:
        ret = rbuffer.store_frame(obs)
        net_obs = rbuffer.encode_recent_observation().astype(np.float32)
        action = model(Variable(torch.from_numpy(net_obs).to(args.device).float().unsqueeze(0)/255.0)).data.max(1)[1].cpu().numpy()
        obs, reward, done, info = env.step(int(action))
        r += reward
        rbuffer.store_effect(ret, int(action), reward, done) 
    print(r)
