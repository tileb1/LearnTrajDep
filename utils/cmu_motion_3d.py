from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import torch
from utils.model import IdentityAutoencoder


class CMU_Motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=10, output_n=10, split=0, data_mean=0, data_std=0, dim_used=0,
                 dct_used=15, autoencoder=IdentityAutoencoder()):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = dct_used

        actions = data_utils.define_actions_cmu(actions)

        if split == 0:
            path_to_data = path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = path_to_data + '/test/'
            is_test = True

        all_seqs, dim_ignore, dim_use, data_mean, data_std = data_utils.load_data_cmu_3d(path_to_data, actions,
                                                                                     input_n, output_n,
                                                                                     data_std=data_std,
                                                                                     data_mean=data_mean,
                                                                                     is_test=is_test)
        if not is_test:
            dim_used = dim_use

        self.all_seqs = all_seqs
        self.dim_used = dim_used

        # (nb_total_seq, len_seq, nb_joints)
        all_seqs = torch.from_numpy(all_seqs[:, :, dim_used]).float()

        # (nb_total_seq, nb_joints, hidden_dim)
        self.all_seqs_encoded = autoencoder(all_seqs.transpose(2, 1))[1]
        tmp = all_seqs.transpose(2, 1).clone()

        # Pad with last seen skeleton
        tmp[:, :, input_n:] = tmp[:, :, input_n - 1, None]
        self.all_seqs_encoded_padded = autoencoder(tmp)[1]

        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        return self.all_seqs_encoded.shape[0]

    def __getitem__(self, item):
        return self.all_seqs_encoded_padded[item], self.all_seqs_encoded[item], \
               self.all_seqs[item]
