from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import os
from model import DLANET
from dataloader import CarlaDataset
from torch.utils.data import DataLoader

def get_acc(output, target):
    score = float(torch.sum(output == target)) / float(output.shape[0])
    return float(score) * 100

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    model = DLANET()
    model = nn.DataParallel(model).cuda().eval()

    dataset = CarlaDataset(path=os.path.join('..', 'imitation_data1'))
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=True,
        num_workers=8,
        shuffle=True
    )

    total_steps = 0
    for epoch in range(10, 11):
        model.load_state_dict(torch.load(os.path.join('trained_models', 'epoch_%d.pth' % epoch)))
        cnt = 0
        total_acc = 0.0
        for batch_i, (img, mask, action) in enumerate(data_loader):
            cnt += 1
            print("Training epoch [{:3d}/{:3d}], batch [{:3d}/{:3d}]".format(epoch + 1, 100, batch_i + 1, len(data_loader)))
            img = img.cuda()
            mask = mask.cuda()
            action = action.cuda()
            with torch.no_grad():
                logit, _ = model(img)
            pred = torch.argmax(logit, dim=1).long()
            total_acc += get_acc(pred, action)
            with open('output.txt', 'a') as f:
                for i in range(logit.shape[0]):
                    f.write(str(int(pred[i]))+' '+str(int(action[i])) +'\n')

            total_steps += 1
        total_acc /= cnt
        with open('eval_log.txt', 'a') as f:
            f.write('Epoch %d Accuracy %0.2f\n' % (epoch, total_acc))
