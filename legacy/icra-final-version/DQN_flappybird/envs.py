import gym
from atari_wrappers import *
from gym.spaces.box import Box
#import py_TORCS
import os
torcs_path = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))
# from torcs_wrapper import *
import gym_gridworld

def create_atari_env(env_id, seed=0, deepmind_wrap=False, game_config='michigan.xml', isServer=1,
    continuous=False, args=None):
    if env_id == 'TORCS-v0':
        env = gym.make('TORCS-v0')
        config = os.path.join(torcs_path, 'pyTORCS/py_TORCS/py_TORCS/game_config/'+game_config)
        env.init(game_config=config, isServer=isServer, continuous=continuous, resize=False, ID=args.torcs_id)
        env = TorcsWrapper(env, continuous=continuous, args=args)
        obs = env.reset()
    else:
        env = gym.make(env_id)
        if env_id == 'gridworld-v0':
            env.reward_mode = args.reward_mode
            #if args.plan:
            #    env.init_states(args.plan)
            if args.render:
                env.verbose = True
                _ = env.reset()
        if deepmind_wrap:
            env = wrap_deepmind(env)
        else:
            env = AtariRescale42x42(env)
            env.seed(seed)
    return env

class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 255.0, [84, 84, 3])

    def _observation(self, observation):
        return _process_frame42(observation)

def _process_frame42(frame):
    frame = cv2.resize(frame, (84, 84)) 
    frame = frame.astype(np.float32)
    return frame
