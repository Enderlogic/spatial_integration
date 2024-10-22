# Base MMVAEplus class definition
import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import get_mean

class MMVAEplus(nn.Module):
    """
    MMVAEplus class definition.
    """
    def __init__(self, params, *vaes):
        super(MMVAEplus, self).__init__()
        self.pp = Normal # Prior distribution
        self.ps = Normal
        self.vaes = nn.ModuleList([vae(params) for vae in vaes]) # List of unimodal VAEs (one for each modality)
        self.modelName = None  # Filled-in in subclass
        self.params = params # Model parameters (i.e. args passed to main script)

    def forward(self, x, K=1):
        """
        Forward function.
        Input:
            - x: list of data samples for each modality
            - K: number of samples for reparameterization in latent space

        Returns:
            - qu_xs: List of encoding distributions (one per encoder)
            - px_us: Matrix of self- and cross- reconstructions. px_zs[m][n] contains
                    m --> n  reconstruction.
            - uss: List of latent codes, one for each modality. uss[m] contains latents inferred
                   from modality m. Note there latents are the concatenation of private and shared latents.
        """
        qu_xs, uss = [], []
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # Loop over unimodal vaes
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(x[m], K=K) # Get Encoding dist, Decoding dist, Latents for unimodal VAE m modality
            qu_xs.append(qu_x) # Append encoding distribution to list
            uss.append(us) # Append latents to list
            px_us[m][m] = px_u  # Fill-in self-reconstructions in the matrix
        # Loop over unimodal vaes and compute cross-modal reconstructions
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal with cross-modal reconstructions
                    # Get shared latents from encoding modality e
                    _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                    # Resample modality-specific encoding from modality-specific auxiliary distribution for decoding modality m
                    pw = vae.pw(*vae.pw_params_aux)
                    latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                    # Fixed for cuda (sorry)
                    if not self.params.no_cuda and torch.cuda.is_available():
                        latents_w.cuda()
                    # Combine shared and resampled private latents
                    us_combined = torch.cat((latents_w, z_e), dim=-1)
                    # Get cross-reconstruction likelihood
                    px_us[e][d] = vae.px_u(*vae.dec(us_combined))
        return qu_xs, px_us, uss


    def generate_unconditional(self, N):
        """
        Unconditional generation.
        Args:
            N: Number of samples to generate.
        Returns:
            Generations
        """
        with torch.no_grad():
            data = []
            # Sample N shared latents
            pz = self.pz(*self.pz_params)
            latents_z = pz.rsample(torch.Size([N]))
            # Decode for all modalities
            for d, vae in enumerate(self.vaes):
                pw = self.pw(*self.pw_params)
                latents_w = pw.rsample([latents_z.size()[0]])
                latents = torch.cat((latents_w, latents_z), dim=-1)
                px_u = vae.px_u(*vae.dec(latents))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality


    # def reconstruct(self, data):
    #     """
    #     Test-time reconstruction
    #     Args:
    #         data: Input
    #
    #     Returns:
    #         Reconstructions
    #     """
    #     with torch.no_grad():
    #         _, px_zs, _ = self.forward(data)
    #         # cross-modal matrix of reconstructions
    #         recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
    #     return recons

    def self_and_cross_modal_generation_forward(self, data, K=1):
        """
        Test-time self- and cross-model generation forward function.
        Args:
            data: Input

        Returns:
            Unimodal encoding distribution, Matrix of self- and cross-modal reconstruction distrubutions, Latent embeddings

        """
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(data[m], K=K)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        for e, us in enumerate(uss):
            latents_w, latents_z = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
            for d, vae in enumerate(self.vaes):
                mean_w, scale_w = self.pw_params
                # Tune modality-specific std prior
                # scale_w = factor * scale_w
                pw = self.pw(mean_w, scale_w)
                latents_w_new = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                us_new = torch.cat((latents_w_new, latents_z), dim=-1)
                if e != d:  # fill-in off-diagonal
                    px_us[e][d] = vae.px_u(*vae.dec(us_new))
        return qu_xs, px_us, uss

    def self_and_cross_modal_generation(self, data):
        """
        Test-time self- and cross-reconstruction.
        Args:
            data: Input

        Returns:
            Matrix of self- and cross-modal reconstructions

        """
        with torch.no_grad():
            _, px_us, _ = self.self_and_cross_modal_generation_forward(data)
            # ------------------------------------------------
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_u) for px_u in r] for r in px_us]
        return recons
