from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import torch
from utils.model import IdentityAutoencoder


class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2,
                 autoencoder=IdentityAutoencoder(), subset=True):
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

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        acts = data_utils.define_actions(actions)

        if subset:
            subs = np.array([[1], [5], [11]])
            acts = ['walking']

        subjs = subs[split]

        all_data, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 (input_n + output_n) * 2)

        ############################################### NO DOWNSAMPLING ###############################################
        self.all_seqs1 = all_data[:, input_n:2 * input_n + output_n, :]
        self.dim_used = dim_used

        # (nb_total_seq, len_seq, nb_joints)
        all_seqs1 = torch.from_numpy(self.all_seqs1[:, :, dim_used]).float()

        # (nb_total_seq, nb_joints, hidden_dim)
        self.all_seqs_encoded1 = autoencoder(all_seqs1.transpose(2, 1))[1]

        # Pad with last seen skeleton
        tmp1 = all_seqs1.transpose(2, 1).clone()
        tmp1[:, :, input_n:] = tmp1[:, :, input_n - 1, None]
        self.all_seqs_encoded_padded1 = autoencoder(tmp1)[1]

        ################################################# DOWNSAMPLING ################################################
        self.all_seqs2 = all_data[:, ::2, :]  # TODO: check if correct

        # (nb_total_seq, len_seq, nb_joints)
        all_seqs2 = torch.from_numpy(self.all_seqs2[:, :, dim_used]).float()

        # (nb_total_seq, nb_joints, hidden_dim)
        self.all_seqs_encoded2 = autoencoder(all_seqs2.transpose(2, 1))[1]

        # Pad with last seen skeleton
        tmp2 = all_seqs2.transpose(2, 1).clone()
        tmp2[:, :, input_n:] = tmp2[:, :, input_n - 1, None]
        self.all_seqs_encoded_padded2 = autoencoder(tmp2)[1]

    def __len__(self):
        return self.all_seqs_encoded1.shape[0]

    def __getitem__(self, item):
        return self.all_seqs_encoded_padded1[item], self.all_seqs_encoded1[item], \
               self.all_seqs1[item], self.all_seqs_encoded_padded2[item], self.all_seqs_encoded2[item], \
               self.all_seqs2[item]


class H36motion3DRaw(H36motion3D):
    def __init__(self, path_to_data, actions, input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2,
                 autoencoder=IdentityAutoencoder(), subset=False):
        super().__init__(path_to_data, actions, input_n=input_n, output_n=output_n, dct_used=dct_used,
                         split=split, sample_rate=sample_rate, autoencoder=autoencoder, subset=subset)

    def __getitem__(self, item):
        return self.all_seqs_encoded_padded1[item], self.all_seqs_encoded1[item], self.all_seqs1[item]


class H36motion3DSubsampled(H36motion3D):
    def __init__(self, path_to_data, actions, input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2,
                 autoencoder=IdentityAutoencoder(), subset=False):

        super().__init__(path_to_data, actions, input_n=input_n, output_n=output_n, dct_used=dct_used,
                         split=split, sample_rate=sample_rate, autoencoder=autoencoder, subset=subset)

    def __getitem__(self, item):
        return self.all_seqs_encoded_padded2[item], self.all_seqs_encoded2[item], self.all_seqs2[item]