import os
import time
from collections import deque
import pickle
import pdb
from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import cv2

def train(env, actor, critic, param_noise, action_noise, memory, tau=0.01, eval_env=None, param_noise_adaption_interval=50, env_id=0, args=None):
    rank = MPI.COMM_WORLD.Get_rank()

    nb_epochs = args.nb_epochs
    nb_epoch_cycles = args.nb_epoch_cycles
    render_eval = args.render_eval
    reward_scale = args.reward_scale
    render = args.render
    normalize_returns = args.normalize_returns
    normalize_observations = args.normalize_observations
    critic_l2_reg = args.critic_l2_reg
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    popart = args.popart
    gamma = args.gamma
    clip_norm = args.clip_norm
    nb_train_steps = args.nb_train_steps
    nb_rollout_steps = args.nb_rollout_steps
    nb_eval_steps = args.nb_eval_steps
    batch_size = args.batch_size

    max_action = np.array([1.0, 1.0])#env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, (256, 256, 3), (2,),
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        if os.path.exists("/tmp/ddpg_model_%d.ckpt" % env_id):
            saver.restore(sess, "/tmp/ddpg_model_%d.ckpt" % env_id)
        if os.path.exists("/tmp/ddpg_episode_log_%d.txt" % env_id):
            t = int(open("/tmp/ddpg_episode_log_%d.txt" % env_id).readlines()[-1].split(' ')[1])
            print(t)
        else:
            t = 0
        sess.graph.finalize()

        agent.reset()
        obs, info = env.reset()
        obs = cv2.resize(obs, (256, 256))
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        done_cnt = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    obs = cv2.resize(obs, (256, 256))
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    action = np.clip(action, -1, 1)
                    new_obs, r, done, info = env.step(action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    new_obs = cv2.resize(new_obs, (256,  256))
                    episode_reward += r['without_pos']
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r['without_pos'], new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        done_cnt += 1
                        with open('/tmp/ddpg_episode_log_%d.txt' % env_id, 'a') as f:
                            f.write('step %d reward %0.4f\n' % (t, episode_reward))
                        print('episode ', episodes, ' reward is', episode_reward)
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1


                        agent.reset()
                        obs, info = env.reset()


                        # Evaluate
                        if done_cnt % 5 == 0:
                            done = False
                            eval_obs, info = env.reset()
                            eval_rewards = 0
                            while not done:
                                eval_obs = cv2.resize(eval_obs, (256, 256))
                                eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                                eval_action = np.clip(eval_action, -1, 1)
                                eval_obs, eval_r, done, info = env.step(eval_action)
                                eval_rewards += eval_r['without_pos']
                            with open('ddpg_eval_log_%d.txt' % env_id, 'a') as f:
                                f.write('step %d reward %0.4f\n' % (t, eval_rewards))
                            print('step ', t, ' evalreward ', eval_rewards)
                        agent.reset()
                        obs, info = env.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()


            mpi_size = MPI.COMM_WORLD.Get_size()
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            save_path = saver.save(sess, "/tmp/ddpg_model_%d.ckpt" % env_id)
            print("DDPG Model saved in path: %s" % save_path)
