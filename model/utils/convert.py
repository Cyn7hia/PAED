'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
'''


import torch


def to_var(x, on_cpu=False, gpu_id=None):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id)
        #x = Variable(x)
    return x


def to_tensor(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data
