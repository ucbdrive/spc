def turn_off_grad(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

def test(args, model, up, predictor, further, env):
    turn_off_grad(model)
    turn_off_grad(predictor)
    turn_off_grad(further)
    '''
        inputs = torch.autograd.Variable(torch.ones(1, 9, 480, 640), requires_grad = False)
        obs = env.reset()
        obs, reward, done, info = env.step(1)
        true_obs = np.repeat((obs.transpose(2, 0, 1) - obs_avg) / obs_std, 3, axis = 0)
        for i in range(300):
            inputs[:] = torch.from_numpy(true_obs)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output, feature_map = model(inputs)
            pos, angle, speed = further(feature_map)
            print(pos.data.cpu().numpy() * 7, angle.data.cpu().numpy(), speed.data.cpu().numpy())
            print(info['trackPos'], info['angle'], info['speed'])
            print()
            action = naive_driver({'trackPos': pos.data.cpu().numpy() * 7, 'angle':angle.data.cpu().numpy()})

            obs, reward, done, info = env.step(action)
            obs = (obs.transpose(2, 0, 1) - obs_avg) / obs_std
            true_obs = np.concatenate((true_obs[3:], obs), axis = 0)
    '''
    env.close()
