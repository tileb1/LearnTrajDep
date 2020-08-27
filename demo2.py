#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import torch
import torch.optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion3d import H36motion3D
import utils.model as nnmodel
import utils.data_utils as data_utils
import utils.viz as viz
from utils.old_model import GCNDCTAdapter


def main(opt):
    # is_cuda = torch.cuda.is_available()

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    sample_rate = opt.sample_rate

    # My model
    model = nnmodel.InceptionGCN(opt.linear_size, opt.dropout, num_stage=opt.num_stage, node_n=66, opt=opt)
    model_path_len = './checkpoint/test/ckpt_main_3d_3D_in10_out25_dct_n_30_best.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len, map_location='cpu')
    err_best = ckpt['err']
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # # Mao's model
    mao_model = GCNDCTAdapter(opt.dct_n, opt.linear_size, opt.dropout, num_stage=opt.num_stage, node_n=66)
    model_path_len = './checkpoint/pretrained/h36m3D_in10_out25_dctn30.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len, map_location='cpu')
    err_best = ckpt['err']
    start_epoch = ckpt['epoch']
    mao_model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion3D(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                                   sample_rate=sample_rate)
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
        print(act)
        for i, (_, inputs, all_seq) in enumerate(test_data[act]):
            print(act)
            preds = model(inputs)
            pred_exmap = all_seq.clone()
            pred_exmap[:, :, dim_used] = preds.detach().transpose(1, 2)

            # mao
            preds_mao = mao_model(all_seq, dim_used)
            pred_exmap_mao = all_seq.clone()
            pred_exmap_mao[:, :, dim_used] = preds_mao.detach().transpose(1, 2)

            for k in range(0, 1):
                plt.cla()
                figure_title = "action:{}, seq:{},".format(act, (k + 1))
                viz.plot_predictions2(pred_exmap.numpy()[k, :, :], pred_exmap.numpy()[k, :, :], fig, ax, figure_title,
                                      mao_pred=pred_exmap_mao.numpy()[k, :, :], action=act, gt=False)
                # plt.pause(1)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
