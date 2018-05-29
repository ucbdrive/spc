import torch
import torch.nn as nn
import numpy as np

def test_mat(mat):
    mat = mat/255.0
    return mat

a = np.ones(3)#torch.rand(3)
print(a)
b = test_mat(a)
print(a)
