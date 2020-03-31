from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import torch


class H36motion(Dataset):

    def __init__(self, path_to_data, actions, input_n=10, output_n=10, split=0, sample_rate=2, data_mean=0,
                 data_std=0, autoencoder=lambda x: x):
        """
        read h36m data to get the dct coefficients.
        :param path_to_data:
        :param actions: actions to read
        :param input_n: past frame length
        :param output_n: future frame length
        :param dct_n: number of dct coeff. used
        :param split: 0 train, 1 test, 2 validation
        :param sample_rate: 2
        :param data_mean: mean of expmap
        :param data_std: standard deviation of expmap
        """

        self.path_to_data = path_to_data
        self.split = split
        self.autoencoder = autoencoder
        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])

        acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        subjs = subs[split]
        all_seqs, dim_ignore, dim_use, data_mean, data_std = data_utils.load_data(path_to_data, subjs, acts,
                                                                                  sample_rate,
                                                                                  input_n + output_n,
                                                                                  data_mean=data_mean,
                                                                                  data_std=data_std,
                                                                                  input_n=input_n)

        self.data_mean = data_mean
        self.data_std = data_std

        # first 6 elements are global translation and global rotation
        dim_used = dim_use[6:]  # TODO: indices in angle space
        self.all_seqs = all_seqs
        self.dim_used = dim_used

        # (nb_total_seq, len_seq, nb_joints)
        all_seqs = torch.from_numpy(all_seqs[:, :, dim_used]).float()

        # (nb_total_seq, nb_joints, hidden_dim)
        self.all_seqs_encoded = autoencoder(all_seqs.transpose(2, 1))
        tmp = all_seqs.transpose(2, 1).clone()

        # Pad with last seen skeleton
        tmp[:, :, input_n:] = tmp[:, :, input_n-1, None]
        self.all_seqs_encoded_padded = autoencoder(tmp)

        # all_seqs = all_seqs.transpose(0, 2, 1)  # TODO: change index in sequence with index in angle space
        # all_seqs = all_seqs.reshape(-1, input_n + output_n)  # TODO
        # all_seqs = all_seqs.transpose()
        # dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
        # dct_m_out, _ = data_utils.get_dct_matrix(input_n + output_n)
        #
        # # padding the observed sequence so that it has the same length as observed + future sequence
        # pad_idx = np.repeat([input_n - 1], output_n)  # TODO: repeat last index of input, output_n times
        # i_idx = np.append(np.arange(0, input_n), pad_idx)
        #
        # input_dct_seq = np.matmul(dct_m_in[:dct_n, :], all_seqs[i_idx, :])
        # input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_n])
        #
        # output_dct_seq = np.matmul(dct_m_out[:dct_n], all_seqs)
        # output_dct_seq = output_dct_seq.transpose().reshape([-1, len(dim_used), dct_n])
        #
        # self.input_dct_seq = input_dct_seq
        # self.output_dct_seq = output_dct_seq

    def __len__(self):
        return np.shape(self.all_seqs_encoded)[0]

    def __getitem__(self, item):
        return self.all_seqs_encoded_padded[item], self.all_seqs_encoded[item], \
               self.all_seqs[item]
