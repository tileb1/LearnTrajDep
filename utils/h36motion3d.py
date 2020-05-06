from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import torch
from utils.model import IdentityAutoencoder


class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2,
                 autoencoder=IdentityAutoencoder(), subset=False):
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
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 input_n + output_n)
        self.all_seqs = all_seqs[:, input_n:]
        self.dim_used = dim_used

        # (nb_total_seq, len_seq, nb_joints)
        all_seqs = torch.from_numpy(all_seqs[:, :, dim_used]).float()

        # # (nb_total_seq, nb_joints, hidden_dim)
        # self.all_seqs_encoded = autoencoder(all_seqs.transpose(2, 1))[1]
        # tmp = all_seqs.transpose(2, 1).clone()
        #
        # # Pad with last seen skeleton
        # tmp[:, :, input_n:] = tmp[:, :, input_n-1, None]
        # self.all_seqs_padded = tmp
        # self.all_seqs_encoded_padded = autoencoder(tmp)[1]

        self.input = all_seqs.transpose(2, 1)[:, :, :input_n]
        self.output = all_seqs.transpose(2, 1)[:, :, input_n:]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, item):
        return self.input[item], self.output[item], self.all_seqs[item]
