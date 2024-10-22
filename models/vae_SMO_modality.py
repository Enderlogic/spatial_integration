import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.distributions import Normal
from torch.nn import Module
from torch_geometric.nn import GATv2Conv, InnerProductDecoder

from .utils import Constants, get_mean


class VAE(nn.Module):
    """
    Unimodal VAE class. M unimodal VAEs are then used to construct a mixture-of-experts multimodal VAE.
    """

    def __init__(self, enc, dec, params):
        super(VAE, self).__init__()
        self.pp = Normal  # Prior distribution class (private latent)
        self.ps = Normal
        self.enc = enc  # Encoder object
        self.dec = dec  # Decoder object
        self.modelName = None  # Model name : defined in subclass
        self.params = params  # Parameters (i.e. args passed to the main script)
        self._pzp_params_aux = None  # defined in subclass
        self._qu_x_params = None  # Parameters of posterior distributions: populated in forward
        self.llik_scaling = 1.0  # Likelihood scaling factor for each modality

    @property
    def pzp_params_aux(self):
        """Handled in multimodal VAE subclass, depends on the distribution class"""
        return self._pzp_params_aux

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
        self._qu_x_params = self.enc(x)  # Get encoding distribution params from encoder
        qu_x = self.qu_x(*self._qu_x_params)  # Encoding distribution
        us = qu_x.rsample(torch.Size([K]))  # K-sample reparameterization trick
        px_u = self.px_u(*self.dec(us))  # Get decoding distribution
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


class Encoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_dim, hidden_dim, s_dim, p_dim, heads=1):
        super(Encoder, self).__init__()
        self.s_dim = s_dim
        self.p_dim = p_dim
        self.conv_s1 = GATv2Conv(in_dim, hidden_dim, heads=heads)
        self.conv_s2 = GATv2Conv(hidden_dim * heads, s_dim * 2, heads=1, concat=False)

        self.conv_p1 = GATv2Conv(in_dim, hidden_dim, heads=heads)
        self.conv_p2 = GATv2Conv(hidden_dim * heads, p_dim * 2, heads=1, concat=False)

    def forward(self, x, edge_index):
        s = F.relu(self.conv_s1(x, edge_index))
        s = self.conv_s2(s, edge_index)

        p = self.conv_p1(x, edge_index)
        p = self.conv_p2(p, edge_index)
        return torch.cat((s[:, :self.s_dim], p[:, :self.p_dim]), dim=-1), torch.cat((F.softplus(
            s[:, self.s_dim:]).squeeze() + Constants.eta, F.softplus(p[:, self.p_dim:]).squeeze() + Constants.eta),
                                                                                    dim=-1)


class SMO_SRT(VAE):
    """ Unimodal VAE subclass for SRT modality """

    def __init__(self, params):
        super(SMO_SRT, self).__init__(
            Encoder(params.srt_dim, params.hidden_srt_dim, params.zs_dim, params.zp_dim),  # Encoder model
            InnerProductDecoder(),  # Decoder model
            params  # Params (args passed to main)
        )

        self._pzp_params_aux = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.zp_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.zp_dim), requires_grad=True)
            # It is important that this log-variance vector is learnable (see paper)
        ])

        self.modelName = 'srt'
        self.llik_scaling = 1.
        self.params = params

    @property
    def pzp_params_aux(self):
        """

        Returns: Parameters of prior auxiliary distribution for modality-specific latent code

        """
        return self._pzp_params_aux[0], F.softplus(self._pzp_params_aux[1]) + Constants.eta


class SMO_SPR(VAE):
    """ Unimodal VAE subclass for SRT modality """

    def __init__(self, params):
        super(SMO_SPR, self).__init__(
            Encoder(params.spr_dim, params.hidden_spr_dim, params.zs_dim, params.zp_dim),  # Encoder model
            InnerProductDecoder(),  # Decoder model
            params  # Params (args passed to main)
        )

        self._pzp_params_aux = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.zp_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.zp_dim), requires_grad=True)
            # It is important that this log-variance vector is learnable (see paper)
        ])

        self.modelName = 'spr'
        self.llik_scaling = 1.
        self.params = params

    @property
    def pzp_params_aux(self):
        """

        Returns: Parameters of prior auxiliary distribution for modality-specific latent code

        """
        return self._pzp_params_aux[0], F.softplus(self._pzp_params_aux[1]) + Constants.eta
