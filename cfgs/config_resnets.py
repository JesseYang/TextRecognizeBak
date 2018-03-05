import numpy as np
from easydict import EasyDict as edict

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

cfg = edict()

cfg.name = 'data'

#================================
# Input Part
#================================
cfg.input = edict()
cfg.input.height = 30
cfg.input.width = None
cfg.input.channel = 1
cfg.dictionary = [" ", "\"", "$", "%", "&", "'", "(", ")", "*",
                  "-", ".", "/", "0", "1", "2", "3", "4", "5",
                  "6", "7", "8", "9", ":", "<", ">", "?", "[",
                  "]", "a", "b", "c", "d", "e", "f", "g", "h",
                  "i", "j", "k", "l", "m", "n", "o", "p", "q",
                  "r", "s", "t", "u", "v", "w", "x", "y", "z",
                  "{", "}", 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                  'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                  'Z', ';', ',', '!', '=', '|', '`', '_', '#', '~', '+']
cfg.label_size = len(cfg.dictionary) + 1
cfg.train_list = [cfg.name + "_train.txt"]
cfg.test_list = cfg.name + "_test.txt"


cfg.learning_rate = [(0, 1e-5), (3, 3e-5), (6, 6e-5), (10, 1e-4), (60, 1e-5)]

cfg.augmentors = [
            imgaug.ToFloat32(),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False)]),
            imgaug.Clip(),
            imgaug.ToUint8(),
        ]

#================================
# ResNets Part
#================================
cfg.resnets = edict()
cfg.resnets.depth = 18 # should be one of `{18, 34, 50, 101}`

#================================
# RNN Part
#================================
cfg.rnn = edict()
cfg.rnn.hidden_size = 660
cfg.rnn.hidden_layers_no = 2

#================================
# Train Part
#================================
cfg.weight_decay = 5e-4
