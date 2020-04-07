import torch

MY_DEVICE = 'cpu'
is_cuda = torch.cuda.is_available() and False

if is_cuda:
    MY_DEVICE = 'cuda'