from flappy_bird_wrapper import flappy_bird_wrapper
from dqn_agent import *
from envs import *
import argparse
from dqn_utils import *
import pdb
import sys
from lidar_utils import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--save-path', type=str, default='models')
parser.add_argument('--log-name', type=str, default='log_ours_adv.txt')
parser.add_argument('--load-old-q-value', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--frame-history-len', type=int, default=4)
parser.add_argument('--env-id', type=str, default='torcs-v0')
parser.add_argument('--buffer-size', type=int, default=100000)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-starts', type=int, default=10000)
parser.add_argument('--learning-freq', type=int, default=10)
parser.add_argument('--target-update-freq', type=int, default=1000)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--epsilon-frames', type=int, default=100000)
parser.add_argument('--wrap-deepmind', action='store_true')
parser.add_argument('--test-dqn', action='store_true')
parser.add_argument('--change-model', action='store_true')
parser.add_argument('--game-config', type=str, default='michigan.xml')
parser.add_argument('--isServer', type=int, default=1)
parser.add_argument('--continuous', action='store_true')
parser.add_argument('--use-segmentation', action='store_true')
parser.add_argument('--torcs-id', type=int, default=6)
parser.add_argument('--render', action='store_true')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--plan', type=str)
parser.add_argument('--reward-mode', type=int, default=0)


def train(env, dqn_agent, args, num):
    last_obs, _ = env.reset()
    last_obs = cv2.resize(last_obs, (80, 80))
    last_obs = last_obs.transpose(2, 0, 1)
    rewards, epi_rewards = 0, []
    test_rewards = []
    for t in range(num, 2000000):
        action = dqn_agent.sample_action(last_obs, t)
        last_obs, reward, done, info = env.step(action)
        last_obs = cv2.resize(last_obs, (80, 80))
        last_obs = last_obs.transpose(2, 0, 1)
        rewards += reward

        print('Round %d, action %d reward %d' % (t, action, reward), end='\r')

        if done:
            with open('reward_mode_%d.txt' % args.reward_mode, 'a') as f:
                f.write('Step %d Reward %0.2f\n' % (t, rewards))
            last_obs, _ = env.reset()
            last_obs = cv2.resize(last_obs, (80, 80))
            last_obs = last_obs.transpose(2, 0, 1)
            epi_rewards.append(rewards)
            rewards = 0.0
            done = False
            test_reward = 0
            test_steps = 0
            while not done and test_steps < 49:
                action = dqn_agent.sample_action(last_obs, 1e7)
                last_obs, reward, done, info = env.step(action)
                last_obs = cv2.resize(last_obs, (80, 80))
                last_obs = last_obs.transpose(2, 0, 1)
                test_reward += reward
                test_steps += 1
                # dqn_agent.store_effect(action, reward, done)
            test_rewards.append(test_reward)
            last_obs, _ = env.reset()
            last_obs = cv2.resize(last_obs, (80, 80))
            last_obs = last_obs.transpose(2, 0, 1)
            if len(epi_rewards) % 10 == 0:
                print('episode ', len(epi_rewards), ' t ', t, ' reward ', np.mean(test_rewards[-10:]))
                with open(os.path.join(args.save_path, args.log_name), 'a') as f:
                    f.write('episode '+str(len(epi_rewards))+' step '+str(t)+' reward '+str(np.mean(test_rewards[-50:]))+'\n')
        dqn_agent.store_effect(action, reward, done)
        
        if t % args.learning_freq == 0 and dqn_agent.replay_buffer.can_sample(args.batch_size) and t > args.learning_starts:
            dqn_agent.train_model(args.batch_size, t)
    
    return dqn_agent

def test(env, dqn_agent, args):
    last_obs, _ = env.reset()
    last_obs = cv2.resize(last_obs, (80, 80))
    last_obs = last_obs.transpose(2, 0, 1)
    rewards, epi_rewards = 0, []
    while len(epi_rewards) < 10:
        action = dqn_agent.sample_action(last_obs, 1000000)
        last_obs, reward, done, info = env.step(action)
        last_obs = cv2.resize(last_obs, (80, 80))
        last_obs = last_obs.transpose(2, 0, 1)
        rewards += reward
        if done:
            last_obs, _ = env.reset()
            last_obs = cv2.resize(last_obs, (80, 80))
            last_obs = last_obs.transpose(2, 0, 1)
            epi_rewards.append(rewards)
            rewards = 0.0
        dqn_agent.store_effect(action, reward, done)
    print('episode reward is ', np.mean(epi_rewards))
    with open(os.path.join(args.save_path, args.log_name), 'a') as f:
        f.write('test reward is '+str(np.mean(epi_rewards))+' std '+str(np.std(epi_rewards))+'\n')
    return
 
if __name__ == '__main__':
    args = parser.parse_args()
    set_global_seeds(args.seed)
    env = flappy_bird_wrapper()
    obs, _ = env.reset()
    obs = obs.transpose(2, 0, 1)
    if args.test_dqn == False:
        exploration = PiecewiseSchedule([
            (0, 1),
            (args.epsilon_frames, 0.00)
            ], outside_value=0.00
        )
    else:
        exploration = PiecewiseSchedule([
            (0, 0),
            (10000, 0.0)
            ], outside_value = 0.0
        )
    
    if args.env_id != 'lidargrid':
        img_c, img_h, img_w = obs.shape
        print('observation space shape is', img_h, img_w, img_c)
        num_actions = env.action_space.n
        dqn_agent = DQNAgent(img_c, args.frame_history_len, num_actions, 
                lr = args.learning_rate, 
                exploration=exploration, 
                save_path=args.save_path,
                device=args.device,
                img_h=img_h, img_w=img_w)
    else:
        dqn_agent = DQNAgent(8, 1, 5, lr=args.learning_rate, exploration=exploration,
                    save_path=args.save_path,
                    device=args.device,
                    img_h=1, img_w=1)
    if args.load_old_q_value or args.test_dqn:
        num = dqn_agent.load_model()
    else:
        num = 0
    
    if args.test_dqn == False:
        train(env, dqn_agent, args, num)
    else:
        test(env, dqn_agent, args)
