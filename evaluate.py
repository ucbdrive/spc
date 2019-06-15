from __future__ import division, print_function
from manager import BufferManager, ActionSampleManager
from utils import generate_guide_grid, log_frame, record_screen, draw_from_pred, from_variable_to_numpy
from models import init_models
import os
import sys
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import multiprocessing as _mp
mp = _mp.get_context('spawn')


def draw(action, step, obs_var, net, args, action_var, name):
    if not os.path.isdir(os.path.join('demo', str(step), name)):
        os.makedirs(os.path.join('demo', str(step), name))
    s = 'Next Action: [%0.1f, %0.1f]\n' % (action[0][0], action[0][1])
    # cv2.putText(raw_obs, 'Next Action: [%0.1f, %0.1f]' % (action[0], action[1]), (70, 400), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)

    action = torch.from_numpy(action).view(1, args.pred_step, args.num_total_act)
    action = Variable(action.cuda().float(), requires_grad=False)
    obs_var = obs_var / 255.0
    obs_var = obs_var.view(1, 1, 9, args.frame_height, args.frame_width)
    with torch.no_grad():
        output = net(obs_var, action, training=False, action_var=action_var)
        output['offroad_prob'] = F.softmax(output['offroad_prob'], -1)
        output['coll_prob'] = F.softmax(output['coll_prob'], -1)

    for i in range(args.pred_step):
        img = draw_from_pred(args, from_variable_to_numpy(torch.argmax(output['seg_pred'][0, i+1], 0)))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.putText(img, 'OffRoad: %0.2f%%' % float(100*output['offroad_prob'][0, i, 1]), (20, 160), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
        # cv2.putText(img, 'Collision: %0.2f%%' % float(100*output['coll_prob'][0, i, 1]), (20, 200), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
        s += 'Step %d\n' % i
        s += 'OffRoad: %0.2f%%\n' % float(100*output['offroad_prob'][0, i, 1])
        s += 'Collision: %0.2f%%\n' % float(100*output['coll_prob'][0, i, 1])
        # s += 'Distance: %0.2f%%\n' % float(output['dist'][0, i, 0])
        cv2.imwrite(os.path.join('demo', str(step), name, 'seg%d.png' % (i+1)), img)

    with open(os.path.join('demo', str(step), name, 'log.txt'), 'w') as f:
        f.write(s)


def evaluate_policy(args, env):
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

    print('Start episode...')

    for step in range(1000):
        obs_var = buffer_manager.store_frame(obs, info)
        action, guide_action = action_manager.sample_action(net=net,
                                                            obs=obs,
                                                            obs_var=obs_var,
                                                            action_var=action_var,
                                                            exploration=exploration,
                                                            step=step,
                                                            explore=False,
                                                            testing=True)
        draw(action, step, obs_var, net, args, action_var, 'outcome')
        cv2.imwrite(os.path.join('demo', str(step), 'obs.png'), cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
        action = action[0]
        obs, reward, done, info = env.step(action)

        action_var = buffer_manager.store_effect(guide_action=guide_action,
                                                 action=action[0],
                                                 reward=reward,
                                                 done=done,
                                                 collision=info['collision'],
                                                 offroad=info['offroad'])
        if args.recording:
            log_frame(obs, action[0], video_folder, video)

        if done:
            print('Episode finished ...')
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
                else:
                    signal.value = 0
                    p.join()
                    del p
            break
