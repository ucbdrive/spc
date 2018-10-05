from utils import *
preddata = PredData('data/data', 'data/action','data/done', 'data/coll', 4, 9)

loader = DataLoader(dataset=preddata, batch_size=13)
for x in loader:
    x = x
    break

import pdb
pdb.set_trace()
