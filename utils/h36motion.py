from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import torch


def get_raw_loader(loader):
    for i in loader:
        yield i[0]['raw'], i[1]['raw'], i[2]


class H36motion(Dataset):

    def __init__(self, path_to_data, actions, input_n=10, output_n=10, split=0, sample_rate=2, data_mean=0,
                 data_std=0, autoencoder=lambda x: (1, x)):
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

        subs = np.array([[1], [5], [11]])
        acts = ['walking']

        subjs = subs[split]
        all_seqs, dim_ignore, dim_use, data_mean, data_std = data_utils.load_data(path_to_data, subjs, acts,
                                                                                  sample_rate,
                                                                                  input_n + output_n,
                                                                                  data_mean=data_mean,
                                                                                  data_std=data_std,
                                                                                  input_n=input_n,
                                                                                  preprocess=False)

        all_seqs_smoothed, _, _, _, _ = data_utils.load_data(path_to_data, subjs, acts,
                                                             sample_rate,
                                                             input_n + output_n,
                                                             data_mean=None,
                                                             data_std=None,
                                                             input_n=input_n,
                                                             preprocess=True)

        self.data_mean = data_mean
        self.data_std = data_std

        # first 6 elements are global translation and global rotation
        dim_used = dim_use[6:]  # TODO: indices in angle space
        self.all_seqs = all_seqs
        self.dim_used = dim_used

        # (nb_total_seq, len_seq, nb_joints)
        all_seqs = torch.from_numpy(all_seqs[:, :, dim_used]).float()
        all_seqs_smoothed = torch.from_numpy(all_seqs_smoothed[:, :, dim_used]).float()

        # (nb_total_seq, nb_joints, hidden_dim)
        self.all_seqs_encoded = autoencoder(all_seqs.transpose(2, 1))[1]
        self.all_seqs_smoothed = autoencoder(all_seqs_smoothed.transpose(2, 1))[1]
        tmp = all_seqs.transpose(2, 1).clone()
        tmp_smoothed = all_seqs_smoothed.transpose(2, 1).clone()

        # Pad with last seen skeleton
        tmp[:, :, input_n:] = tmp[:, :, input_n - 1, None]
        tmp_smoothed[:, :, input_n:] = tmp_smoothed[:, :, input_n - 1, None]
        self.all_seqs_encoded_padded = autoencoder(tmp)[1]
        self.all_seqs_smoothed_padded = autoencoder(tmp_smoothed)[1]

    def __len__(self):
        return self.all_seqs_encoded.shape[0]

    def __getitem__(self, item):
        return {'raw': self.all_seqs_encoded_padded[item], 'smooth': self.all_seqs_smoothed_padded[item]}, \
               {'raw': self.all_seqs_encoded[item], 'smooth': self.all_seqs_smoothed[item]}, \
               self.all_seqs[item]
