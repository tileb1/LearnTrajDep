#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd
from matplotlib import pyplot as plt

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion import H36motion
import utils.model as nnmodel
import utils.data_utils as data_utils
import utils.viz as viz
from utils.model import TimeAutoencoder


def main(opt):
    # is_cuda = torch.cuda.is_available()

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    sample_rate = opt.sample_rate

    model = nnmodel.GCN(input_feature=opt.dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=48)
    # if is_cuda:
    #     model.cuda()
    # model_path_len = './checkpoint/pretrained/h36m_in10_out10_dctn20.pth.tar'
    model_path_len = './checkpoint/test/ckpt_main_in10_out25_dctn30_last.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    # if is_cuda:
    #     ckpt = torch.load(model_path_len)
    # else:

    ckpt = torch.load(model_path_len, map_location='cpu')
    err_best = ckpt['err']
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    time_autoencoder = TimeAutoencoder(opt.input_n + opt.output_n, opt.dct_n)
    utils.load_model(time_autoencoder, 'autoencoder_{}_{}.pt'.format(opt.input_n + opt.output_n, opt.dct_n))

    # data loading
    print(">>> loading data")
    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                                 sample_rate=sample_rate, autoencoder=time_autoencoder)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    dim_used = test_dataset.dim_used
    print(">>> data loaded !")

    model.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    for act in acts:
        for i, (inputs, _, all_seq) in enumerate(test_data[act]):
            outputs = model(inputs)
            preds = time_autoencoder.decoder(outputs)
            pred_exmap = all_seq.clone()
            pred_exmap[:, :, dim_used] = preds.detach().transpose(1, 2)

            for k in range(8):
                plt.cla()
                figure_title = "action:{}, seq:{},".format(act, (k + 1))
                viz.plot_predictions(all_seq.numpy()[k, :, :], pred_exmap.numpy()[k, :, :], fig, ax, figure_title)
                plt.pause(1)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
