import torch
import socket

MY_DEVICE = 'cpu'
is_cuda = torch.cuda.is_available() and socket.gethostname() != 'ThinkPad-W550s'

if is_cuda:
    MY_DEVICE = 'cuda'
