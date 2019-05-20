from __future__ import division, print_function
from manager import BufferManager, ActionSampleManager
from utils import generate_guide_grid, train_model, train_guide_action, log_frame, color_text, record_screen
from models import init_models
import os
import sys
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl
import multiprocessing as _mp
mp = _mp.get_context('spawn')


def train_policy(args, env, max_steps=40000000):
    guides = generate_guide_grid(args.bin_divide)
    train_net, net, optimizer, epoch, exploration, num_steps = init_models(args)

    buffer_manager = BufferManager(args)
    action_manager = ActionSampleManager(args, guides)
    action_var = Variable(torch.from_numpy(np.array([-1.0, 0.0])).repeat(1, args.frame_history_len - 1, 1), requires_grad=False).float()

    # prepare video recording
    if args.recording:
        video_folder = os.path.join(args.video_folder, "%d" % num_steps)
        os.makedirs(video_folder, exist_ok=True)
        if args.sync:
            video = cv2.VideoWriter(os.path.join(video_folder, 'video.avi'),
                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                    24.0, (args.frame_width, args.frame_height), True)
        else:
            video = None
            signal = mp.Value('i', 1)
            p = mp.Process(target=record_screen,
                           args=(signal,
                                 os.path.join(video_folder, 'video.avi'),
                                 1280, 800, 24))
            p.start()

    # initialize environment
    obs, info = env.reset()
    if args.recording:
        log_frame(obs, buffer_manager.prev_act, video_folder, video)

    num_episode = 1
    print('Start training...')

    for step in range(num_steps, max_steps):
        obs_var = buffer_manager.store_frame(obs, info)
        action, guide_action = action_manager.sample_action(net=net,
                                                            obs=obs,
                                                            obs_var=obs_var,
                                                            action_var=action_var,
                                                            exploration=exploration,
                                                            step=step,
                                                            explore=num_episode % 2)
        obs, reward, done, info = env.step(action)
        print("action [{0:.2f}, {1:.2f}]".format(action[0], action[1]) + " " +
              "collision {}".format(str(bool(info['collision']))) + " " +
              "off-road {}".format(str(bool(info['offroad']))) + " " +
              "speed {0:.2f}".format(info['speed']) + " " +
              "reward {0:.2f}".format(reward) + " " +
              "explore {0:.2f}".format(exploration.value(step))
              )

        action_var = buffer_manager.store_effect(guide_action=guide_action,
                                                 action=action,
                                                 reward=reward,
                                                 done=done,
                                                 collision=info['collision'],
                                                 offroad=info['offroad'])
        if args.recording:
            log_frame(obs, action, video_folder, video)

        if done:
            print('Episode {} finished'.format(num_episode))
            if not args.sync and args.recording:
                signal.value = 0
                p.join()
                del p

        # train SPN
        if buffer_manager.spc_buffer.can_sample(args.batch_size) and ((not args.sync and done) or (args.sync and step % args.learning_freq == 0)):
            # train model
            for ep in range(args.num_train_steps):
                optimizer.zero_grad()
                loss = train_model(args=args,
                                   net=train_net,
                                   spc_buffer=buffer_manager.spc_buffer)
                if args.use_guidance:
                    loss += train_guide_action(args=args,
                                               net=train_net,
                                               spc_buffer=buffer_manager.spc_buffer,
                                               guides=guides)
                print('loss = %0.4f\n' % loss.data.cpu().numpy())
                loss.backward()
                optimizer.step()
                epoch += 1
            net.load_state_dict(train_net.state_dict())

            # save model
            if epoch % args.save_freq == 0:
                print(color_text('Saving models ...', 'green'))
                torch.save(train_net.module.state_dict(),
                           os.path.join(args.save_path, 'model', 'pred_model_%09d.pt' % step))
                torch.save(optimizer.state_dict(),
                           os.path.join(args.save_path, 'optimizer', 'optimizer.pt'))
                with open(os.path.join(args.save_path, 'epoch.pkl'), 'wb') as f:
                    pkl.dump(epoch, f)
                buffer_manager.save_spc_buffer()
                print(color_text('Model saved successfully!', 'green'))

        if done:
            # reset video recording
            if args.recording:
                if args.sync:
                    video.release()
                    if sys.platform == 'linux':  # save memory
                        os.system('ffmpeg -y -i {0} {1}'.format(
                            os.path.join(video_folder, 'video.avi'),
                            os.path.join(video_folder, 'video.mp4')
                        ))
                        if os.path.exists(os.path.join(video_folder, 'video.mp4')):
                            os.remove(os.path.join(video_folder, 'video.avi'))

                    video_folder = os.path.join(args.video_folder, "%d" % step)
                    os.makedirs(video_folder, exist_ok=True)
                    video = cv2.VideoWriter(os.path.join(video_folder, 'video.avi'),
                                            cv2.VideoWriter_fourcc(*'MJPG'),
                                            24.0, (args.frame_width, args.frame_height), True)
                else:
                    video_folder = os.path.join(args.video_folder, "%d" % step)
                    os.makedirs(video_folder, exist_ok=True)

                    signal.value = 1
                    p = mp.Process(target=record_screen,
                                   args=(signal, os.path.join(video_folder, 'obs.avi'), 1280, 800, 24))
                    p.start()

            num_episode += 1
            obs, info = env.reset()
            buffer_manager.reset(step)
            action_manager.reset()
            if args.recording:
                log_frame(obs, buffer_manager.prev_act, video_folder, video)
