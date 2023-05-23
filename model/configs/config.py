'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
'''
from torch import optim
import torch.nn as nn
from model.layers.rnncells import StackedLSTMCell, StackedGRUCell

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
rnncell_dict = {'lstm': StackedLSTMCell, 'gru': StackedGRUCell}


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                if key == 'rnncell':
                    value = rnncell_dict[value]
                setattr(self, key, value)


def get_config(config):
    for key in config.keys():
        config[key] = Config(**config[key])

    return Config(**config)