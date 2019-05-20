from __future__ import division, print_function

class make_env(object):
    def __init__(self, args):
        super(make_env, self).__init__()
        self.args = args

    def __enter__(self):
        if 'torcs' in self.args.env:
            import gym
            from .TORCS.torcs_wrapper import TorcsWrapper
            self.env = gym.make('TORCS-v0')
            self.env.init(isServer=self.args.server,
                          continuous=True,
                          resize=not self.args.eval,
                          ID=self.args.id)
            return TorcsWrapper(self.env)

        elif 'carla' in self.args.env:
            from carla.client import make_carla_client
            from .CARLA.carla_env import CarlaEnv
            with make_carla_client('localhost', 2019) as client:
                return CarlaEnv(client)

        elif 'gta' in self.args.env:
            from .GTAV.gta_env import GtaEnv
            from .GTAV.gta_wrapper import GTAWrapper
            return GTAWrapper(GtaEnv(autodrive=None))


    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'torcs' in self.args.env:
            self.env.close()

        elif 'carla' in self.args.env:
            pass  # self.client.__exit__(exc_type, exc_val, exc_tb)

        elif 'gta' in self.args.env:
            pass

