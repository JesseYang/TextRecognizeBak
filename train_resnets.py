#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from collections import Counter
import operator
import six
from six.moves import map, range
import json
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from cfgs.config import cfg
from reader import Data, CTCBatchData
from mapper import *

class RecogResult(Inferencer):
    def __init__(self, names):
        if not isinstance(names, list):
            self.names = [names]
        else:
            self.names = names
        self.mapper = Mapper()

    def _get_output_tensors(self):
        return self.names

    def _before_inference(self):
        self.results = []

    def _datapoint(self, output):
        for prediction in output[0]:
            line = self.mapper.decode_output(prediction)
            self.results.append(line)

    def _after_inference(self):
        for idx, line in enumerate(self.results):
            print(str(idx) + ": " + line)
        return {}

class Model(ModelDesc):

    def __init__(self):
        pass
        # self.batch_size = batch_size

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.input.height, None, cfg.input.channel], 'feat'),   # bxmaxseqx39
                InputDesc(tf.int64, None, 'labelidx'),  # label is b x maxlen, sparse
                InputDesc(tf.int32, None, 'labelvalue'),
                InputDesc(tf.int64, None, 'labelshape'),
                InputDesc(tf.int32, [None], 'seqlen'),   # b
                ]

    # def _build_graph(self, input_vars):
    def _build_graph(self, inputs):
        with tf.device('/gpu:1'):
            l, labelidx, labelvalue, labelshape, seqlen = inputs
            tf.summary.image('input_img', l)
            label = tf.SparseTensor(labelidx, labelvalue, labelshape)
            l = tf.cast(l, tf.float32)
            l = l / 255.0 * 2 - 1

            self.batch_size = tf.shape(l)[0]
        #================================
        # ResNets Part
        #================================
        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            nonlocal seqlen
            ch_in = l.get_shape().as_list()[-1] 
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            print(seqlen)
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            # padding = 'VALID'
            # seqlen = tf.add(tf.round(tf.divide(tf.subtract(seqlen, 3), stride)), 1)

            # padding = 'SAME'
            seqlen = tf.ceil(tf.divide(seqlen, stride))

            l = Conv2D('conv2', l, ch_out, 3)
            # padding = 'VALID'
            # seqlen = tf.add(tf.subtract(seqlen, 3), 1)

            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[-1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            nonlocal seqlen
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l
        net_cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = net_cfg[cfg.resnets.depth]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
            # padding = 'VALID'
            # seqlen = tf.add(tf.round(tf.divide(tf.subtract(seqlen, 3), 1)), 1)
            # seqlen = tf.add(tf.round(tf.divide(tf.subtract(seqlen, 2), 2)), 1)

            # padding = 'SAME'
            # seqlen = tf.ceil(tf.divide(seqlen, 2))
            # seqlen = tf.ceil(tf.divide(seqlen, 2))
            l = (LinearWrap(l)
                      .apply(layer, 'group0', block_func, 32, defs[0], 1, first=True)
                      .apply(layer, 'group1', block_func, 64, defs[1], 1)
                      .apply(layer, 'group2', block_func, 128, defs[2], 1)
                      .apply(layer, 'group3', block_func, 256, defs[3], 1)
                      .BNReLU('bnlast')())
        seqlen = tf.cast(seqlen, tf.int32)
        out_channel = 256
        feature_height = l.get_shape().as_list()[1]
        feature_size = feature_height * out_channel


        #================================
        # RNN Part
        #================================
        l = tf.transpose(l, perm=[0, 2, 1, 3])
        l = tf.reshape(l, [self.batch_size, -1, feature_size])

        if cfg.rnn.hidden_layers_no > 0:
            cell_fw = [tf.nn.rnn_cell.BasicLSTMCell(cfg.rnn.hidden_size) for _ in range(cfg.rnn.hidden_layers_no)]
            cell_bw = [tf.nn.rnn_cell.BasicLSTMCell(cfg.rnn.hidden_size) for _ in range(cfg.rnn.hidden_layers_no)]
            l = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, l, dtype=tf.float32)
            feature_size = cfg.rnn.hidden_size


        #================================
        # FC Part
        #================================
        l = tf.reshape(l[0], [-1, 2 * feature_size])
        # l = tf.reshape(l, [-1, feature_size])
        output = BatchNorm('bn', l)
        logits = FullyConnected('fc', output, cfg.label_size, nl=tf.identity,
                                W_init=tf.truncated_normal_initializer(stddev=0.01))
        logits = tf.reshape(logits, (self.batch_size, -1, cfg.label_size))


        #================================
        # CTC Part
        #================================
        loss = tf.nn.ctc_loss(inputs=logits,
                                labels=label,
                                sequence_length=seqlen,
                                time_major=False)
        self.cost = tf.reduce_mean(loss, name='cost')

        # prediction error
        logits = tf.transpose(logits, [1, 0, 2])

        isTrain = get_current_tower_context().is_training
        predictions = tf.to_int32(tf.nn.ctc_greedy_decoder(inputs=logits,
                                                            sequence_length=seqlen)[0][0])
        # predictions = tf.to_int32(tf.nn.ctc_beam_search_decoder(inputs=logits,
        #                                                    sequence_length=seqlen)[0][0])

        dense_pred = tf.sparse_tensor_to_dense(predictions, name="prediction")

        err = tf.edit_distance(predictions, label, normalize=True)
        err.set_shape([None])
        err = tf.reduce_mean(err, name='error')
        summary.add_moving_summary(err, self.cost)

    def get_gradient_processor(self):
        return [GlobalNormClip(400)]

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 3e-4, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'
    ds = Data(train_or_test, shuffle=isTrain)
    if isTrain:
        augmentors = cfg.augmentors
    else:
        augmentors = []
    ds = AugmentImageComponent(ds, augmentors)
    ds = CTCBatchData(ds, batch_size)
    if isTrain:
        # ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
        ds = PrefetchDataZMQ(ds, 1)
    return ds

def get_config(args):
    ds_train = get_data("train", int(args.batch_size))
    # ds_test = get_data("test", args.batch_size)
    ds_test = get_data("test", 1)

    return TrainConfig(
        dataflow = ds_train,
        callbacks = [
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', cfg.learning_rate),
            # HyperParamSetterWithFunc('learning_rate',
            #                          lambda e, x: x / 1.05 ),
            InferenceRunner(ds_test, [ScalarStats('cost'), RecogResult('prediction')]),
            # StatMonitorParamSetter('learning_rate', 'error',
            #                        lambda x: x * 0.2, 0, 5),
            HumanHyperParamSetter('learning_rate'),
            # PeriodicCallback(
            #     InferenceRunner(ds_test, [ScalarStats('error')]), 1),
        ],
        model = Model(),
        max_epoch = 200,
        steps_per_epoch = 1000
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=0)
    parser.add_argument('--batch_size', help='batch size', default=8)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    config = get_config(args)
    if args.load:
        config.session_init = SaverRestore(args.load)
    QueueInputTrainer(config).train()

