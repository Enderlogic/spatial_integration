# CUB Image-Captions Unimodal VAE Image model specification
import os
import json
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import Constants
from .base_vae import VAE
from .encoder_decoder_blocks.cnn_cub_text import Enc, Dec
from torch.utils.data import DataLoader
from dataset_CUB import CUBSentences

# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590


class CUB_Sentence(VAE):
    """ Unimodal VAE subclass for Text modality CUBICC experiment """

    def __init__(self, params):
        super(CUB_Sentence, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,      # prior
            likelihood_dist=dist.OneHotCategorical,                                             # likelihood
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,       # posterior
            enc=Enc(params.latent_dim_w, params.latent_dim_z, dist=params.priorposterior),      # Encoder model
            dec=Dec(params.latent_dim_w, params.latent_dim_z),                                   # Decoder model
            params=params)                                                                      # Params (args passed to main)
        self._pw_params_aux = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=True)  # It is important that this log-variance vector is learnable (see paper)
        ])

        self.modelName = 'cubC'
        self.llik_scaling = 1.

        self.fn_2i = lambda t: t.cpu().numpy().astype(int)
        self.fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s
        self.vocab_file = params.datadir + '/cub/oc:{}_msl:{}/cub.vocab'.format(minOccur, maxSentLen)

        self.maxSentLen = maxSentLen
        self.vocabSize = vocabSize

        self.i2w = self.load_vocab()
        self.params = params

    @property
    def pw_params_aux(self):
        """

        Returns: Parameters of prior auxiliary distribution for modality-specific latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params_aux[0], F.softplus(self._pw_params_aux[1]) + Constants.eta
        else:
            return self._pw_params_aux[0], F.softmax(self._pw_params_aux[1], dim=-1) * self._pw_params_aux[1].size(-1) + Constants.eta


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = lambda data: torch.Tensor(data)
        t_data = CUBSentences(self.params.datadir, split='train', one_hot=True, transpose=False, transform=tx, max_sequence_length=maxSentLen)
        s_data = CUBSentences(self.params.datadir, split='test', one_hot=True, transpose=False, transform=tx, max_sequence_length=maxSentLen)

        train_loader = DataLoader(t_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(s_data, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader


    def load_vocab(self):
        # call dataloader function to create vocab file
        if not os.path.exists(self.vocab_file):
            _, _ = self.getDataLoaders(256)
        with open(self.vocab_file, 'r') as vocab_file:
            vocab = json.load(vocab_file)
        return vocab['i2w']


    
