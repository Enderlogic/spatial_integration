# Base VAE class definition

# Imports
import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import get_mean


class VAE(nn.Module):
    """
    Unimodal VAE class. M unimodal VAEs are then used to construct a mixture-of-experts multimodal VAE.
    """
    def __init__(self, enc, dec, params):
        super(VAE, self).__init__()
        self.pp = Normal # Prior distribution class (private latent)
        self.ps = Normal
        self.enc = enc # Encoder object
        self.dec = dec # Decoder object
        self.modelName = None # Model name : defined in subclass
        self.params = params # Parameters (i.e. args passed to the main script)
        self._pp_params_aux = None # defined in subclass
        self._qu_x_params = None  # Parameters of posterior distributions: populated in forward
        self.llik_scaling = 1.0 # Likelihood scaling factor for each modality


    @property
    def pp_params_aux(self):
        """Handled in multimodal VAE subclass, depends on the distribution class"""
        return self._pp_params_aux

    @property
    def qu_x_params(self):
        """Get encoding distribution parameters (already adapted for the specific distribution at the end of the Encoder class)"""
        if self._qu_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qu_x_params

    def forward(self, x, K=1):
        """
        Forward function
        Returns:
            Encoding dist, latents, decoding dist

        """
        self._qu_x_params = self.enc(x) # Get encoding distribution params from encoder
        qu_x = self.qu_x(*self._qu_x_params) # Encoding distribution
        us = qu_x.rsample(torch.Size([K])) # K-sample reparameterization trick
        px_u = self.px_u(*self.dec(us)) # Get decoding distribution
        return qu_x, px_u, us

    def reconstruct(self, data):
        """
        Test-time reconstruction.
        """
        with torch.no_grad():
            qu_x = self.qu_x(*self.enc(data))
            latents = qu_x.rsample(torch.Size([1]))  # no dim expansion
            px_u = self.px_u(*self.dec(latents))
            recon = get_mean(px_u)
        return recon
