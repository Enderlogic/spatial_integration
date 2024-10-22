# Train MMVAEplus on CUB Image-Captions dataset
import os
import shutil
import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch
from torch import optim
import models
import models.objectives as objectives
from models.mmvaeplus_SMO import SMO_srt_spr
from models.utils import Logger, Timer, save_model_light
from models.utils import unpack_data_cubIC as unpack_data

parser = argparse.ArgumentParser(description='MMVAEplus')
parser.add_argument('--obj', type=str, default='dreg', choices=['elbo', 'dreg'],
                    help='objective to use')
parser.add_argument('--K', type=int, default=10,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train')
parser.add_argument('--p-dim', type=int, default=32, metavar='L',
                    help='private latent dimensionality (default: 32)')
parser.add_argument('--s-dim', type=int, default=64, metavar='L',
                    help='shared latent dimensionality (default: 64)')
parser.add_argument('--print-freq', type=int, default=50, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--llik_scaling_srt', type=float, default=5.0,
                    help='likelihood scaling factor spatial transcriptomics data')
parser.add_argument('--llik_scaling_spr', type=float, default=5.0,
                    help='likelihood scaling factor spatial proteomics data')
parser.add_argument('--outputdir', type=str, default='../outputs',
                    help='Output directory')
parser.add_argument('--priorposterior', type=str, default='Normal', choices=['Normal', 'Laplace'],
                    help='distribution choice for prior and posterior')

# args
args = parser.parse_args()
args.latent_dim = args.s_dim + args.p_dim

# Random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# CUDA stuff
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

model = SMO_srt_spr(args).to(device)

# Set experiment name if not set
if not args.experiment:
    args.experiment = model.modelName

# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)

# Load CUB Image-Captions
train_loader, test_loader = model.getDataLoaders(args.batch_size,  device=device)
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj)
t_objective = objective


def train(epoch):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        data = unpack_data(dataT, device=device)
        optimizer.zero_grad()
        loss = -objective(model, data, K=args.K)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    # Epoch loss
    epoch_loss = b_loss / len(train_loader.dataset)
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))



if __name__ == '__main__':
    with Timer('MMVAEplus') as t:
        for epoch in range(1, args.epochs + 1):
            train(epoch)

