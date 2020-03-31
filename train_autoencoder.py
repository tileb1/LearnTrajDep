from utils.opt import Options
from utils.h36motion import H36motion
from torch.utils.data import DataLoader
from utils.model import TimeAutoencoder
import torch.optim as optim
import torch.nn as nn
from utils.constants import *
from progress.bar import Bar
import time
from utils.utils import save_model


def train_autoencoder(opt):
    train_dataset = H36motion(path_to_data=opt.data_dir, actions='all', input_n=opt.input_n, output_n=opt.output_n,
                              split=0, sample_rate=opt.sample_rate, autoencoder=lambda x: x)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)

    autoencoder = TimeAutoencoder(opt.input_n + opt.output_n, opt.dct_n)
    autoencoder.train()
    autoencoder.to(MY_DEVICE)
    optimizer = optim.Adam(autoencoder.parameters(), lr=opt.lr_autoencoder)
    loss_function = nn.MSELoss()

    for epoch in range(10):
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
            loss = loss_function(out_true, true_seq)
            loss += loss_function(out_padded, padded_seq)
            average_epoch_loss = (i * average_epoch_loss + loss.item()) / (i+1)
            loss.backward()
            optimizer.step()

            bar.suffix = '{}/{}|batch time: {:.4f}s|total time: {:.2f}s|average loss: {:.4E}'.format(i,
                                        len(train_loader)-1, time.time() - bt, time.time() - st, average_epoch_loss)
            bar.next()

        bar.finish()

    autoencoder.eval()
    save_model(autoencoder, 'autoencoder_' + str(opt.input_n + opt.output_n) + '_' + str(opt.dct_n) + '.pt')


if __name__ == "__main__":
    option = Options().parse()
    train_autoencoder(option)
