"""Label smoothing module."""

from typing import OrderedDict

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from AMINO.utils.init_object import init_object
from AMINO.utils.tensor import dim_merge

IGNORE_ID = -1
TORCH_PI = torch.acos(torch.zeros(1)).item() * 2

class LABEL_SMOOTHING_LOSS(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        smoothing (float): smoothing rate (0.0 means the conventional CE)
    """
    def __init__(
        self,
        size: int,
        smoothing: float,
        reduction: str = "none",
    ):
        """Construct an LabelSmoothingLoss object."""
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        if reduction == "none":
            self.reduction = torch.nn.Identity()
        elif reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.
        Args:
            x (torch.Tensor): prediction (batch, class)
            target (torch.Tensor): target onehot (batch, class)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(-1) == self.size
        assert target.size(-1) == self.size
        target = target.to(dtype=torch.float)
        if self.training:
            mask = (target == 0.0)
            target = target.masked_fill(mask, self.smoothing)
            target = target.masked_fill(~mask, self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), target)
        return self.reduction(kl)


class AMINO_LOSSES(nn.Module):
    def __init__(
        self,
        nets,
        weights,
    ):
        super().__init__()
        assert list(nets.keys()) == list(weights.keys()), \
            f"the key of loss weight is not same with the key in loss net"
        self.nets = OrderedDict()
        for k, v in nets.items():
            self.nets[k] = init_object(v)
        self.nets = nn.ModuleDict(self.nets)
        self.weights = dict()
        for k, v in weights.items():
            self.weights[k] = nn.Parameter(torch.tensor(v), requires_grad=False)
        self.weights = nn.ParameterDict(self.weights)

    def forward(self, pred_dict, target_dict, len_dict):
        loss_dict = dict()
        loss_dict["total"] = torch.tensor(
            0.0, device=list(self.weights.values())[0].device,
        )
        for key in pred_dict.keys():
            loss_dict[key] = self.nets[key](
                pred_dict[key], target_dict[key]
            ).sum() / len_dict[key]
            loss_dict["total"] += loss_dict[key] * self.weights[key]
        return loss_dict

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        # l = torch.cholesky(a, False)
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l

    '''
       L = torch.cholesky(A)
       should be replaced with
       L = torch.linalg.cholesky(A)
       and
       U = torch.cholesky(A, upper=True)
       should be replaced with
       U = torch.linalg.cholesky(A).transpose(-2, -1).conj().
       '''

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

class AEGMM_LOSS(nn.Module):
    def __init__(self, lambda_energy, lambda_cov, num_gmm):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.num_gmm = num_gmm
        self.mse_obj = nn.MSELoss()

    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = self.mse_obj(x_hat, x) # output, train_data
        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return loss

    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(-2) - mu)

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.num_gmm):
            cov_k = cov[k] + (
                torch.eye(cov[k].size(-1)) * eps
            ).to(device=cov.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((
                Cholesky.apply(cov_k * (2 * TORCH_PI)).diag().prod()
            ).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(device=cov.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=-1) + eps)
        if sample_mean == True:
            E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        gamma_sum = torch.sum(gamma, dim=[0, 1, 2])

        # phi = D
        phi = torch.sum(
            gamma, dim=[0, 1, 2], keepdim=True,
        ) / (
            gamma.size(0)*gamma.size(1)*gamma.size(2) 
            + torch.finfo(torch.float32).eps
        )

        # mu = KxD
        # Bx1xNxD BxNxKx1
        mu = torch.sum(
            z.unsqueeze(-2) * gamma.unsqueeze(-1),
            dim=[0, 1, 2],
            keepdim=True,
        )
        mu /= (gamma_sum.unsqueeze(-1) + + torch.finfo(torch.float32).eps)

        z_mu = (z.unsqueeze(-2) - mu) # z:B,C,T,H (combine dimension), mu:(1,1,1,H,)
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=[0, 1, 2])
        cov /= ( gamma_sum.unsqueeze(-1).unsqueeze(-1) + torch.finfo(torch.float32).eps)

        return phi, mu, cov

class PYRO_Differentiable_Loss(nn.Module):
    # http://pyro.ai/examples/custom_objectives.html#A-Lower-Level-Pattern
    def __init__(self, pyro_infer):
        super().__init__()
        self.fn = init_object(pyro_infer).differentiable_loss
    
    def forward(self, model, guide, *args, **kwargs):
        return self.fn(model, guide, *args, **kwargs)
