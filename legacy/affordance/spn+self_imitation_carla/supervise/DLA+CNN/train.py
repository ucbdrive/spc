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
    model = nn.DataParallel(model).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    dataset = CarlaDataset()
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=True,
        num_workers=8,
        shuffle=True
    )

    total_steps = 0
    for epoch in range(20):
        for batch_i, (img, mask, action) in enumerate(data_loader):
            print("Training epoch [{:3d}/{:3d}], batch [{:3d}/{:3d}]".format(epoch + 1, 100, batch_i + 1, len(data_loader)))
            img = img.cuda()
            mask = mask.cuda()
            action = action.cuda()
            logit, seg = model(img)
            seg_loss = nn.CrossEntropyLoss()(seg, mask)
            IL_loss = nn.CrossEntropyLoss()(logit, action)
            print('seg loss %0.4f, IL loss %0.4f' % (seg_loss.data.cpu().numpy(), IL_loss.data.cpu().numpy()))
            loss = seg_loss + IL_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = get_acc(torch.argmax(logit, dim=1).long(), action)
            with open('DLA_log.txt', 'a') as f:
                print('epoch %d, batch %d, accuracy %0.2f\n' % (epoch, batch_i, acc))
                f.write('epoch %d, batch %d, accuracy %0.2f\n' % (epoch, batch_i, acc))

            total_steps += 1

            if (total_steps % 20) == 0:
                print('Saving model...')
                if not os.path.isdir('trained_models'):
                    os.makedirs('trained_models')
                torch.save(model.state_dict(), os.path.join('trained_models', 'epoch_%d.pth' % epoch))
        scheduler.step()
