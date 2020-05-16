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
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion3d import H36motion3D
import utils.model as nnmodel
import utils.data_utils as data_utils
from utils.constants import *
from utils.model import TimeAutoencoder


def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr

    # save option in log
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_3D_in{:d}_out{:d}_dct_n_{:d}'.format(opt.input_n, opt.output_n, opt.dct_n)

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    time_autoencoder1 = TimeAutoencoder(opt.input_n + opt.output_n, dct_n)
    extension = 'RAW'
    name1 = 'autoencoder_' + str(opt.input_n + opt.output_n) + '_' + str(opt.dct_n) + '_' + extension + '.pt'
    utils.load_model(time_autoencoder1, name1)

    time_autoencoder2 = TimeAutoencoder(opt.input_n + opt.output_n, dct_n)
    extension = 'DOWNSAMPLED'
    name2 = 'autoencoder_' + str(opt.input_n + opt.output_n) + '_' + str(opt.dct_n) + '_' + extension + '.pt'
    utils.load_model(time_autoencoder2, name2)

    model = nnmodel.MultipleGCN(dct_n, opt.linear_size, opt.dropout, time_autoencoder1, time_autoencoder2, opt,
                        num_stage=opt.num_stage, node_n=66)

    model.to(MY_DEVICE)

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.is_load:
        model_path_len = 'checkpoint/test/' + 'ckpt_' + script_name + '_last.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    train_dataset = H36motion3D(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
                                split=0, dct_used=dct_n, sample_rate=sample_rate)

    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion3D(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                                   sample_rate=sample_rate, dct_used=dct_n)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    val_dataset = H36motion3D(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
                              split=2, dct_used=dct_n, sample_rate=sample_rate)

    # load dadasets for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> test data {}".format(test_dataset.__len__()))
    print(">>> validation data {}".format(val_dataset.__len__()))

    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)

        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l = train(train_loader, model, optimizer, lr_now=lr_now, max_norm=opt.max_norm, is_cuda=is_cuda,
                            dim_used=train_dataset.dim_used, dct_n=dct_n)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        v_3d = val(val_loader, model, is_cuda=is_cuda, dim_used=train_dataset.dim_used, dct_n=dct_n)

        ret_log = np.append(ret_log, [v_3d])
        head = np.append(head, ['v_3d'])

        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_l, test_3d = test(test_data[act], model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used, dct_n=dct_n)
            # ret_log = np.append(ret_log, test_l)
            ret_log = np.append(ret_log, test_3d)
            head = np.append(head,
                             [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            if output_n > 10:
                head = np.append(head, [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)

        # update log file and save checkpoint
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if not np.isnan(v_3d):
            is_best = v_3d < err_best
            err_best = min(v_3d, err_best)
        else:
            is_best = False
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)


def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, dim_used=[], dct_n=15):
    t_l = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs1, targets1, all_seq1, inputs2, targets2, all_seq2) in enumerate(train_loader):

        batch_size = inputs1.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()

        # transfer inputs to GPU if needed
        inputs1 = inputs1.to(MY_DEVICE)
        all_seq1 = all_seq1.to(MY_DEVICE)
        inputs2 = inputs2.to(MY_DEVICE)
        all_seq2 = all_seq2.to(MY_DEVICE)

        # forward pass
        optimizer.zero_grad()
        y1, y2, y_final = model(inputs1, inputs2)

        # backward pass
        loss = loss_funcs.my_mpjpe_error_p3d(y_final, all_seq1, dim_used)
        loss += loss_funcs.my_mpjpe_error_p3d(y1, all_seq1, dim_used)
        loss += loss_funcs.my_mpjpe_error_p3d(y2, all_seq2, dim_used)
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        t_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
        break
    bar.finish()
    return lr_now, t_l.avg


def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15):
    N = 0
    t_l = 0
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs1, targets1, all_seq1, inputs2, targets2, all_seq2) in enumerate(train_loader):
        bt = time.time()

        inputs1 = inputs1.to(MY_DEVICE)
        inputs2 = inputs2.to(MY_DEVICE)

        all_seq = all_seq1.to(MY_DEVICE)

        y1, y2, y_final = model(inputs1, inputs2)

        n, seq_len, dim_full_len = all_seq.data.shape

        outputs_3d = y_final.transpose(1, 2)

        pred_3d = all_seq.clone()
        dim_used = np.array(dim_used)

        # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])  # joints not in dim_used
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])  # joints in dim_used
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        pred_3d[:, :, dim_used] = outputs_3d
        pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]

        pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
        targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]

        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            diff = targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3)
            t_3d[k] += torch.mean(torch.norm(diff, 2, 1)).cpu().data.numpy() * n

        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
        break
    bar.finish()
    return t_l / N, t_3d / N


def val(train_loader, model, is_cuda=False, dim_used=[], dct_n=15):
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs1, targets1, all_seq1, inputs2, targets2, all_seq2) in enumerate(train_loader):
        bt = time.time()

        inputs1 = inputs1.to(MY_DEVICE)
        inputs2 = inputs2.to(MY_DEVICE)
        all_seq1 = all_seq1.to(MY_DEVICE)

        y1, y2, y_final = model(inputs1, inputs2)

        n, _, _ = all_seq1.data.shape

        m_err = loss_funcs.my_mpjpe_error_p3d(y_final, all_seq1, dim_used)

        # update the training loss
        t_3d.update(m_err.cpu().data.numpy() * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
        break
    bar.finish()
    return t_3d.avg


if __name__ == "__main__":
    option = Options().parse()
    main(option)
