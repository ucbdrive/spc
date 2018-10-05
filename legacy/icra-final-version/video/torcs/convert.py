import numpy as np
import os
import cv2

def cvt(array):
    res = np.zeros_like(array)

    x, y = np.where(np.all(array == np.array([0, 0, 255]), axis=-1))
    res[x, y] = np.array([70, 70, 70])

    x, y = np.where(np.all(array == np.array([0, 255, 0]), axis=-1))
    res[x, y] = np.array([35, 142, 107])

    x, y = np.where(np.all(array == np.array([0, 0, 0]), axis=-1))
    res[x, y] = np.array([128, 64, 128])

    x, y = np.where(np.all(array == np.array([255, 0, 0]), axis=-1))
    res[x, y] = np.array([180, 130, 70])

    return res

for i in range(1000):
    for j in range(10):
        fname = os.path.join('%d' % i, 'outcome', 'seg%d.png' % (j+1))
        x = cv2.imread(fname)
        x = cvt(x)
        cv2.imwrite(fname, x)