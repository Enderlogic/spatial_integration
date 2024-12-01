import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal, kl_divergence, Distribution, constraints, Gamma, Poisson, \
    MultivariateNormal, HalfCauchy, InverseGamma, LogNormal
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property
from torch.nn import Linear, ParameterList, Sequential, ReLU
from torch.nn.functional import mse_loss
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import tensor, zeros_like, ones_like
from torch_geometric.nn import GATv2Conv, InnerProductDecoder
from torch_geometric.utils import remove_self_loops, negative_sampling
from tornado.gen import multi


def reparameterize(mu, log_var):
    """
    :param mu: mean from the encoder's latent space
    :param log_var: log variance from the encoder's latent space
    """
    std = log_var.exp().sqrt()  # standard deviation
    eps = torch.randn_like(std)  # `randn_like` as we need the same size
    sample = mu + (eps * std)  # sampling
    return sample


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * 1.0
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


# Reference:
# https://github.com/YosefLab/scvi-tools/blob/master/scvi/distributions/_negative_binomial.py
class NegBinom(Distribution):
    """
    Gamma-Poisson mixture approximation of Negative Binomial(mean, dispersion)

    lambda ~ Gamma(mu, theta)
    x ~ Poisson(lambda)
    """
    arg_constraints = {
        'mu': constraints.greater_than_eq(0),
        'theta': constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(self, mu, theta, eps=1e-10):
        """
        Parameters
        ----------
        mu : torch.Tensor
            mean of NegBinom. distribution
            shape - [# genes,]

        theta : torch.Tensor
            dispersion of NegBinom. distribution
            shape - [# genes,]
        """
        self.mu = mu
        self.theta = theta
        self.eps = eps
        super(NegBinom, self).__init__(validate_args=True)

    @torch.inference_mode()
    def sample(self, sample_shape) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = Poisson(l_train).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
        return counts

    def _gamma(self):
        return _gamma(self.theta, self.mu)

    def log_prob(self, x):
        """log-likelihood"""
        ll = torch.lgamma(x + self.theta) - \
             torch.lgamma(x + 1) - \
             torch.lgamma(self.theta) + \
             self.theta * (torch.log(self.theta + self.eps) - torch.log(self.theta + self.mu + self.eps)) + \
             x * (torch.log(self.mu + self.eps) - torch.log(self.theta + self.mu + self.eps))

        return ll


class NegativeBinomial(Distribution):
    r"""Negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi_local-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as
    follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}},
       \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0)
    }
    support = constraints.nonnegative_integer

    def __init__(
            self,
            mu: torch.Tensor | None = None,
            theta: torch.Tensor | None = None,
            zi_logits: torch.Tensor | None = None
    ):
        self._eps = 1e-8
        self.mu = mu
        self.theta = theta
        self.zi_logits = zi_logits
        super().__init__()

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    def get_normalized(self, key) -> torch.Tensor:
        if key == "mu":
            return self.mu
        elif key == "scale":
            return self.scale
        else:
            raise ValueError(f"normalized key {key} not recognized")

    @property
    def variance(self) -> torch.Tensor:
        return self.mean + (self.mean ** 2) / self.theta

    def _gamma(self) -> Gamma:
        return _gamma(self.theta, self.mu)

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: "
                f"{self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
                if self.__dict__[p] is not None
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    r"""Zero-inflated negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi_local-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as
    follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}},
       \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    zi_logits
        Logits scale of zero inflation probability.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "zi_logits": constraints.real
    }
    support = constraints.nonnegative_integer

    def __init__(
            self,
            mu: torch.Tensor | None = None,
            theta: torch.Tensor | None = None,
            zi_logits: torch.Tensor | None = None
    ):
        super().__init__(mu=mu,
                         theta=theta,
                         zi_logits=zi_logits
                         )
        self.zi_logits = zi_logits
        self.mu = mu
        self.theta = theta

    @property
    def mean(self) -> torch.Tensor:
        pi = self.zi_probs
        return (1 - pi) * self.mu

    @property
    def variance(self) -> None:
        raise NotImplementedError

    @lazy_property
    def zi_logits(self) -> torch.Tensor:
        """ZI logits."""
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self) -> torch.Tensor:
        return logits_to_probs(self.zi_logits, is_binary=True)

    @torch.inference_mode()
    def sample(
            self,
            sample_shape: torch.Size | tuple | None = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        samp = super().sample(sample_shape=sample_shape)
        is_zero = torch.rand_like(samp) <= self.zi_probs
        samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
        return samp_

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        return log_zinb_positive(value, self.mu, self.theta, self.zi_logits, eps=1e-08)


def log_zinb_positive(
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor,
        eps: float = 1e-8,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a zinb model.

    Parameters
    ----------
    x
        Data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi
        logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps
        numerical stability constant

    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless
    # of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res


def _gamma(theta, mu):
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = Gamma(concentration=concentration, rate=rate)
    return gamma_d


def kl(p, q):
    return torch.mean(p * torch.log(p / q))


class mmvaeplus(Module):
    """\
    Overall encoder.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2.
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    results: a dictionary including representations and modality weights.

    """

    def __init__(self, omics1_dim, omics2_dim, zs_dim, zp_dim, hidden_dim_omics1=256, hidden_dim_omics2=256,
                 device=torch.device('cpu'), heads=1):
        super(mmvaeplus, self).__init__()
        self.zs_dim = zs_dim
        self.zp_dim = zp_dim
        self.latent_dim = zs_dim + zp_dim
        self.all_dim = zs_dim + zp_dim * 2
        self.device = device

        self.pzs = Normal
        self.pzp = Normal
        self.pw = Normal
        self.prior_u = Normal(
            *ParameterList([Parameter(torch.zeros(self.all_dim, device=device), requires_grad=False),
                            Parameter(torch.ones(self.all_dim, device=device), requires_grad=False)]))
        self.prior_s = Normal(
            *ParameterList([Parameter(torch.zeros(self.all_dim, device=device), requires_grad=False),
                            Parameter(torch.ones(self.all_dim, device=device), requires_grad=False)]))
        self.prior_delta1 = Normal(
            *ParameterList([Parameter(torch.zeros(omics1_dim, device=device), requires_grad=False),
                            Parameter(torch.ones(omics1_dim, device=device), requires_grad=False)]))
        self.prior_delta2 = Normal(
            *ParameterList([Parameter(torch.zeros(omics2_dim, device=device), requires_grad=False),
                            Parameter(torch.ones(omics2_dim, device=device), requires_grad=False)]))
        self.prior_tau = Normal(
            *ParameterList([Parameter(torch.zeros(self.latent_dim, device=device), requires_grad=False),
                            Parameter(torch.ones(self.latent_dim, device=device), requires_grad=False)]))
        self.prior_lambda1 = Normal(
            *ParameterList([Parameter(torch.zeros(self.latent_dim * omics1_dim, device=device), requires_grad=False),
                            Parameter(torch.ones(self.latent_dim * omics1_dim, device=device), requires_grad=False)]))
        self.prior_lambda2 = Normal(
            *ParameterList([Parameter(torch.zeros(self.latent_dim * omics2_dim, device=device), requires_grad=False),
                            Parameter(torch.ones(self.latent_dim * omics2_dim, device=device), requires_grad=False)]))
        self.prior_c = Normal(*ParameterList([Parameter(torch.zeros(1, device=device), requires_grad=False),
                                              Parameter(torch.ones(1, device=device), requires_grad=False)]))

        self.qs_m = Parameter(torch.randn(self.all_dim, device=device), requires_grad=True)
        self.qs_logv = Parameter(torch.zeros(self.all_dim, device=device), requires_grad=True)
        self.qu_m = Parameter(torch.randn(self.all_dim, device=device), requires_grad=True)
        self.qu_logv = Parameter(torch.zeros(self.all_dim, device=device), requires_grad=True)

        self.encoder_zs_omics1 = Encoder(omics1_dim, hidden_dim_omics1, zs_dim, heads)
        self.encoder_zp_omics1 = Encoder(omics1_dim, hidden_dim_omics1, zp_dim, heads)

        self.encoder_zs_omics2 = Encoder(omics2_dim, hidden_dim_omics2, zs_dim, heads)
        self.encoder_zp_omics2 = Encoder(omics2_dim, hidden_dim_omics2, zp_dim, heads)

        self.decoder_omics1 = Decoder(self.latent_dim, omics1_dim)
        self.decoder_omics2 = Decoder(self.latent_dim, omics2_dim)

        self.measurer_omics1 = Measurer(self.zp_dim, hidden_dim_omics1, omics1_dim)
        self.measurer_omics2 = Measurer(self.zp_dim, hidden_dim_omics2, omics2_dim)

        self.theta_omics1 = Parameter(torch.randn(omics1_dim, device=device), requires_grad=True)
        self.theta_omics2 = Parameter(torch.randn(omics2_dim, device=device), requires_grad=True)
        self.to(device)

    def inference(self, x_omics1, x_omics2, edge_index_omics1, edge_index_omics2):
        # omics 1
        zs_mu_omics1, zs_logvar_omics1 = torch.split(self.encoder_zs_omics1(x_omics1, edge_index_omics1),
                                                     [self.zs_dim, self.zs_dim], dim=-1)
        zp_mu_omics1, zp_logvar_omics1 = torch.split(self.encoder_zp_omics1(x_omics1, edge_index_omics1),
                                                     [self.zp_dim, self.zp_dim], dim=-1)
        zs_omics1 = reparameterize(zs_mu_omics1, zs_logvar_omics1)
        zp_omics1 = reparameterize(zp_mu_omics1, zp_logvar_omics1)

        # omics 2
        zs_mu_omics2, zs_logvar_omics2 = torch.split(self.encoder_zs_omics2(x_omics2, edge_index_omics2),
                                                     [self.zs_dim, self.zs_dim], dim=-1)
        zp_mu_omics2, zp_logvar_omics2 = torch.split(self.encoder_zp_omics2(x_omics2, edge_index_omics2),
                                                     [self.zp_dim, self.zp_dim], dim=-1)
        zs_omics2 = reparameterize(zs_mu_omics2, zs_logvar_omics2)
        zp_omics2 = reparameterize(zp_mu_omics2, zp_logvar_omics2)

        u = reparameterize(self.qu_m, self.qu_logv)
        s = reparameterize(self.qs_m, self.qs_logv)
        return {"zs_mu_omics1": zs_mu_omics1, "zs_logvar_omics1": zs_logvar_omics1, "zp_mu_omics1": zp_mu_omics1,
                "zp_logvar_omics1": zp_logvar_omics1, "zs_omics1": zs_omics1, "zp_omics1": zp_omics1,
                "zs_mu_omics2": zs_mu_omics2, "zs_logvar_omics2": zs_logvar_omics2, "zp_mu_omics2": zp_mu_omics2,
                "zp_logvar_omics2": zp_logvar_omics2, "zs_omics2": zs_omics2, "zp_omics2": zp_omics2, "u": u, "s": s}

    def generative(self, inference_outputs):
        # self modality reconstruction
        zs_omics1, zp_omics1 = inference_outputs["zs_omics1"], inference_outputs["zp_omics1"]
        zs_omics2, zp_omics2 = inference_outputs["zs_omics2"], inference_outputs["zp_omics2"]
        z_for_omics1 = F.softmax(torch.cat([zp_omics1, (zs_omics1.detach() + zs_omics2) / 2, zp_omics2], dim=1))[:,
                       :self.latent_dim]
        z_for_omics2 = F.softmax(torch.cat([zp_omics1, (zs_omics1 + zs_omics2.detach()) / 2, zp_omics2], dim=1))[:,
                       self.zp_dim:]
        x_omics1_outputs = self.decoder_omics1(z_for_omics1 / z_for_omics1.sum(1).unsqueeze(1))
        x_omics2_outputs = self.decoder_omics2(z_for_omics2 / z_for_omics2.sum(1).unsqueeze(1))

        return x_omics1_outputs, x_omics2_outputs

    def loss(self, x_omics1, x_omics2, edge_index_omics1, edge_index_omics2):
        inference_outputs = self.inference(x_omics1, x_omics2, edge_index_omics1, edge_index_omics2)
        zs_mu_omics1, zs_logvar_omics1, zp_mu_omics1, zp_logvar_omics1, zs_omics1, zp_omics1 = inference_outputs[
            "zs_mu_omics1"], inference_outputs["zs_logvar_omics1"], inference_outputs["zp_mu_omics1"], \
            inference_outputs["zp_logvar_omics1"], inference_outputs["zs_omics1"], inference_outputs["zp_omics1"]

        zs_mu_omics2, zs_logvar_omics2, zp_mu_omics2, zp_logvar_omics2, zs_omics2, zp_omics2 = inference_outputs[
            "zs_mu_omics2"], inference_outputs["zs_logvar_omics2"], inference_outputs["zp_mu_omics2"], \
            inference_outputs["zp_logvar_omics2"], inference_outputs["zs_omics2"], inference_outputs["zp_omics2"]

        u, s = inference_outputs["u"], inference_outputs["s"]
        x_omics1_output, x_omics2_output = self.generative(inference_outputs)

        # reconstruction loss
        # omics 1
        recon_omics1 = -NegBinom(x_omics1_output["mean"] * x_omics1.sum(1)[:, None], self.theta_omics1.exp()).log_prob(
            x_omics1).sum(1).mean()

        # omics 2
        recon_omics2 = -NegBinom(x_omics2_output["mean"] * x_omics2.sum(1)[:, None], self.theta_omics2.exp()).log_prob(
            x_omics2).sum(1).mean()

        # kl loss
        cov_matrix = u[:, None] * u[:, None].T + torch.diag(s.exp())
        p_z = MultivariateNormal(torch.zeros(self.all_dim, requires_grad=False, device=self.device), cov_matrix)
        q_z = Normal(torch.cat([zp_mu_omics1, (zs_mu_omics1 + zs_mu_omics2) / 2, zp_mu_omics2], dim=1),
                     (torch.cat([zp_logvar_omics1, torch.log(zs_logvar_omics1.exp() / 4 + zs_logvar_omics2.exp() / 4),
                                 zp_logvar_omics2], dim=1) / 2).exp())
        z = torch.cat([zp_omics1, (zs_omics1 + zs_omics2) / 2, zp_omics2], dim=1)
        kl_z = q_z.log_prob(z).sum(1).mean() - p_z.log_prob(z).mean()
        kl_u = kl_divergence(Normal(self.qu_m, (self.qu_logv / 2).exp()), self.prior_u).mean()
        kl_s = kl_divergence(Normal(self.qs_m, (self.qs_logv / 2).exp()), self.prior_s).mean()
        kl_delta = kl_divergence(Normal(x_omics1_output['delta_m'], (x_omics1_output['delta_logv'] / 2).exp()),
                                 self.prior_delta1).mean() + kl_divergence(
            Normal(x_omics2_output['delta_m'], (x_omics2_output['delta_logv'] / 2).exp()), self.prior_delta2).mean()
        kl_tau = kl_divergence(Normal(x_omics1_output['tau_m'], (x_omics1_output['tau_logv'] / 2).exp()),
                               self.prior_tau).mean() + kl_divergence(
            Normal(x_omics2_output['tau_m'], (x_omics2_output['tau_logv'] / 2).exp()), self.prior_tau).mean()
        kl_lambda = kl_divergence(Normal(x_omics1_output['lambda_m'], (x_omics1_output['lambda_logv'] / 2).exp()),
                                  self.prior_lambda1).mean() + kl_divergence(
            Normal(x_omics2_output['lambda_m'], (x_omics2_output['lambda_logv'] / 2).exp()), self.prior_lambda2).mean()
        kl_c = kl_divergence(Normal(x_omics1_output['c_m'], (x_omics1_output['c_logv'] / 2).exp()),
                             self.prior_c).mean() + kl_divergence(
            Normal(x_omics2_output['c_m'], (x_omics2_output['c_logv'] / 2).exp()), self.prior_c).mean()
        lambda_1 = (torch.matmul(x_omics1_output['tau'] ** 2, x_omics1_output['delta'] ** 2) * x_omics1_output[
            'lambd'] ** 2).flatten()
        p_w1 = Normal(torch.zeros_like(x_omics1_output['w_m']),
                      (x_omics1_output['c'] ** 2 * lambda_1 / (x_omics1_output['c'] ** 2 + lambda_1)).sqrt())
        lambda_2 = (torch.matmul(x_omics2_output['tau'] ** 2, x_omics2_output['delta'] ** 2) * x_omics2_output[
            'lambd'] ** 2).flatten()
        p_w2 = Normal(torch.zeros_like(x_omics2_output['w_m']),
                      (x_omics2_output['c'] ** 2 * lambda_2 / (x_omics2_output['c'] ** 2 + lambda_2)).sqrt())
        kl_w = kl_divergence(Normal(x_omics1_output['w_m'], (x_omics1_output['w_logv'] / 2).exp()),
                             p_w1).mean() + kl_divergence(
            Normal(x_omics2_output['w_m'], (x_omics2_output['w_logv'] / 2).exp()), p_w2).mean()

        # measurement loss
        x_omics1_measure = self.measurer_omics1(F.softmax(z, dim=1)[:, self.latent_dim:])
        x_omics2_measure = self.measurer_omics2(F.softmax(z, dim=1)[:, : self.zp_dim])
        var_omics1_measure = (x_omics1_measure.std(1) * x_omics1.sum(1)).mean()
        var_omics2_measure = (x_omics2_measure.std(1) * x_omics2.sum(1)).mean()
        losses = {"recon_omics1": recon_omics1, "recon_omics2": recon_omics2, "kl_z": kl_z, "kl_u": kl_u, "kl_s": kl_s,
                  "kl_delta": kl_delta, "kl_tau": kl_tau, "kl_lambda": kl_lambda, "kl_c": kl_c, "kl_w": kl_w,
                  "var_omics1_measure": var_omics1_measure, "var_omics2_measure": var_omics2_measure}
        return losses

    def measurer_loss(self, x_omics1, x_omics2, edge_index_omics1, edge_index_omics2):
        with torch.no_grad():
            inference_outputs = self.inference(x_omics1, x_omics2, edge_index_omics1, edge_index_omics2)
        zp_omics1 = inference_outputs["zp_omics1"]
        zp_omics2 = inference_outputs["zp_omics2"]

        x_omics1_measure = self.measurer_omics1(zp_omics2)
        x_omics2_measure = self.measurer_omics2(zp_omics1)

        recon_omics1_measure = -NegBinom(x_omics1_measure * x_omics1.sum(1)[:, None],
                                         self.theta_omics1.detach().clone().exp()).log_prob(x_omics1).sum(1).mean()
        recon_omics2_measure = -NegBinom(x_omics2_measure * x_omics2.sum(1)[:, None],
                                         self.theta_omics2.detach().clone().exp()).log_prob(x_omics2).sum(1).mean()
        return {"recon_omics1_measure": recon_omics1_measure, "recon_omics2_measure": recon_omics2_measure}


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

    def __init__(self, in_dim, hidden_dim, out_dim, heads=1):
        super(Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, out_dim * 2, heads=heads, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class Decoder(Module):
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

    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.delta_m = Parameter(torch.ones(out_dim), requires_grad=True)
        self.delta_logv = Parameter(torch.log(torch.ones(out_dim) * .01), requires_grad=True)
        self.tau_m = Parameter(torch.ones(in_dim), requires_grad=True)
        self.tau_logv = Parameter(torch.log(torch.ones(in_dim) * .01), requires_grad=True)
        self.lambda_m = Parameter(torch.ones(in_dim * out_dim), requires_grad=True)
        self.lambda_logv = Parameter(torch.log(torch.ones(in_dim * out_dim) * .01), requires_grad=True)
        self.c_m = Parameter(torch.randn(1), requires_grad=True)
        self.c_logv = Parameter(torch.zeros(1), requires_grad=True)
        self.w_m = Parameter(torch.randn(in_dim * out_dim), requires_grad=True)
        self.w_logv = Parameter(torch.zeros(in_dim * out_dim), requires_grad=True)

    def forward(self, x):
        delta = reparameterize(self.delta_m, self.delta_logv)[None, :]
        tau = reparameterize(self.tau_m, self.tau_logv)[:, None]
        lambd = reparameterize(self.lambda_m, self.lambda_logv).reshape((x.shape[1], -1))
        c = reparameterize(self.c_m, self.c_logv)
        w = reparameterize(self.w_m, self.w_logv).reshape((x.shape[1], -1))
        outputs = {"mean": torch.matmul(x, F.softmax(w, dim=1)), "delta": delta, "tau": tau, "lambd": lambd, "c": c,
                   "w": w, "delta_m": self.delta_m, "delta_logv": self.delta_logv, "tau_m": self.tau_m,
                   "tau_logv": self.tau_logv, "lambda_m": self.lambda_m, "lambda_logv": self.lambda_logv,
                   "c_m": self.c_m, "c_logv": self.c_logv, "w_m": self.w_m, "w_logv": self.w_logv}
        return outputs


class Measurer(Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Measurer, self).__init__()
        self.l1 = Linear(in_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return F.softmax(self.l2(x))
