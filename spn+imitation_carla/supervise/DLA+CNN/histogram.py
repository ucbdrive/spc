import numpy as np
import pandas as pd
from dataloader import CarlaDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = CarlaDataset()
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=True,
        num_workers=8,
        shuffle=True
    )

    cnt = np.zeros(4)
    for batch_i, (img, mask, action) in enumerate(data_loader):
    	for act in list(action):
    		cnt[act] += 1
    print(cnt)
    df = pd.DataFrame(cnt, index=['forward', 'left', 'right', 'stop'])
    plt.figure(figsize=(10, 7))
    ax = df.plot(kind='bar')
    plt.tight_layout()
    plt.savefig('hist.png', dpi=300)