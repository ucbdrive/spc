import os
import cv2
from utils import sample_action, reset_env

def turn_off_grad(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

def test(args, model, up, predictor, further, env):
    turn_off_grad(model)
    turn_off_grad(up)
    turn_off_grad(predictor)
    turn_off_grad(further)
    print('Testing model.')
    v = cv2.VideoWriter('Sample.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 50.0, (256, 256), True)

    obs = reset_env(args, env)
    done = False
    cnt = 0
    while not done:
        cnt += 1
        action = sample_action(args, obs, model, predictor, further)
        obs, reward, done, info = env.step(action)
        bgr = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        obs = (obs.transpose(2, 0, 1) - args.obs_avg) / args.obs_std
        done = done or reward <= -2.5
    env.close()
    v.release()
    os.system('ffmpeg -i Sample.avi Sample.mp4 -y')
    print('Episode length: %d steps.' % cnt)
