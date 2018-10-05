import numpy as np
import cv2

def cvt(array):
    res = np.zeros_like(array)

    x, y = np.where(np.all(array == np.array([0, 0, 255]), axis=-1))
    if len(x) > 0:
        res[x, y] = np.array([70, 70, 70])

    x, y = np.where(np.all(array == np.array([0, 0, 0]), axis=-1))
    if len(x) > 0:
        res[x, y] = np.array([70, 130, 180])

    x, y = np.where(np.all(array == np.array([0, 255, 0]), axis=-1))
    if len(x) > 0:
        res[x, y] = np.array([128, 64, 128])

    return res

def convert(fname):
    x = cv2.imread(fname)
    x = cvt(x)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    cv2.imwrite(fname, x)

for i in range(44):
    for j in range(10):
        convert(str(i) + '/outcome/seg%d.png' % (j+1))