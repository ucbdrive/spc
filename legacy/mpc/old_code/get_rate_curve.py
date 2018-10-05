import numpy as np
import pickle as pkl
import pdb
import os

file_list = sorted(os.listdir('cnfmat'))
for i in range(len(file_list)):
    idx = int(file_list[i].split('_')[-1].split('.')[0])
    if file_list[i][:3] == 'off':
        file_list[i] = 'off_'+str(idx).zfill(9)+'.pkl'
    else:
        file_list[i] = str(idx).zfill(9)+'.pkl'
file_list = sorted(file_list)
coll = []
off = []
for i in range(len(file_list)):
    idx = int(file_list[i].split('_')[-1].split('.')[0])
    if file_list[i][:3] == 'off':
        name = 'cnfmat/off_'+str(idx)+'.pkl'
    else:
        name = 'cnfmat/'+str(idx)+'.pkl'
    try:
        data = pkl.load(open(name,'rb'))
        rate = np.sum(data, 1).reshape((1,2))
        new_rate = rate[0,1]/(rate[0,0]+rate[0,1]+0.0)*1.0
        if file_list[i][:3] == 'off':
            off.append(new_rate)
        else:
            coll.append(new_rate)
    except:
        pass
    
pkl.dump(coll, open('coll.pkl','wb'))
pkl.dump(off, open('off.pkl','wb'))
