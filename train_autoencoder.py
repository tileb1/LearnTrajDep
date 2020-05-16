from utils.opt import Options
from utils.h36motion3d import H36motion3D, H36motion3DRaw, H36motion3DSubsampled
from torch.utils.data import DataLoader
from utils.model import TimeAutoencoder
import torch.optim as optim
import torch.nn as nn
from utils.constants import *
from progress.bar import Bar
import time
from utils.utils import save_model
from utils.model import IdentityAutoencoder


def train_autoencoder(opt, train_loader, extension=''):
    autoencoder = TimeAutoencoder(opt.input_n + opt.output_n, opt.dct_n)
    autoencoder.train()
    autoencoder.to(MY_DEVICE)
    optimizer = optim.Adam(autoencoder.parameters(), lr=opt.lr_autoencoder)
    loss_function = nn.MSELoss()

    for epoch in range(50):
        st = time.time()
        bar = Bar('Epoch {}:'.format(epoch), fill='>', max=len(train_loader))
        average_epoch_loss = 0
        for i, (padded_seq, true_seq, _) in enumerate(train_loader):
            bt = time.time()
            padded_seq = padded_seq.to(MY_DEVICE)
            true_seq = true_seq.to(MY_DEVICE)

            # forward pass
            optimizer.zero_grad()
            out_true, _ = autoencoder(true_seq)
            out_padded, _ = autoencoder(padded_seq)

            # backward pass
            loss = loss_function(out_true, true_seq)  # reconstruction loss
            loss += loss_function(out_padded, padded_seq)  # reconstruction loss

            # loss += 0.5 * loss_function(out_true[:, :, 1:], true_seq[:, :, :-1])  # smoothing loss
            # loss += 0.5 * loss_function(out_true[:, :, :-1], true_seq[:, :, 1:])  # smoothing loss
            # loss += 0.5 * loss_function(out_padded[:, :, 1:], padded_seq[:, :, :-1])  # smoothing loss
            # loss += 0.5 * loss_function(out_padded[:, :, :-1], padded_seq[:, :, 1:])  # smoothing loss

            average_epoch_loss = (i * average_epoch_loss + loss.item()) / (i + 1)
            loss.backward()
            optimizer.step()

            bar.suffix = '{}/{}|batch time: {:.4f}s|total time: {:.2f}s|average loss: {:.4E}'.format(i,
                                                                                                     len(
                                                                                                         train_loader) - 1,
                                                                                                     time.time() - bt,
                                                                                                     time.time() - st,
                                                                                                     average_epoch_loss)
            bar.next()

        bar.finish()

        if epoch % 10 == 9:
            autoencoder.eval()
            save_model(autoencoder, 'autoencoder_' + str(opt.input_n + opt.output_n) + '_' +
                       str(opt.dct_n) + '_' + extension + '.pt')

    autoencoder.eval()
    save_model(autoencoder, 'autoencoder_' + str(opt.input_n + opt.output_n) + '_' +
               str(opt.dct_n) + '_' + extension + '.pt')


if __name__ == "__main__":
    opt = Options().parse()
    dataset = H36motion3DRaw(path_to_data=opt.data_dir, actions='all', input_n=opt.input_n, output_n=opt.output_n,
                             split=0, sample_rate=opt.sample_rate, autoencoder=IdentityAutoencoder(), subset=False)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    train_autoencoder(opt, train_loader, extension='RAW')

    dataset = H36motion3DSubsampled(path_to_data=opt.data_dir, actions='all', input_n=opt.input_n,
                                    output_n=opt.output_n,
                                    split=0, sample_rate=opt.sample_rate, autoencoder=IdentityAutoencoder(),
                                    subset=False)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    train_autoencoder(opt, train_loader, extension='SUBSAMPLED')
