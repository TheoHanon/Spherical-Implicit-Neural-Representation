from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from .third_party.locationencoder.sh import SH
from . import _coords as T


def fourier(
    x: torch.Tensor,
    weight: torch.Tensor,
    omega0: float,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    x:      (..., D)
    weight: (N, D)  (like nn.Linear.weight)
    bias:   (N,) or None
    returns (..., N)
    """
    z = F.linear(x, weight, bias)
    return torch.sin(omega0 * z)


def sph_harm(x: torch.Tensor, l_list: torch.Tensor, m_list: torch.Tensor):
    B = x.size(0)
    N = l_list.numel()
    theta, phi = x[..., 0], x[..., 1]

    outs = torch.empty((B, N), device=x.device, dtype=x.dtype)
    for i in range(N):
        l = int(l_list[i].item())
        m = int(m_list[i].item())
        outs[..., i] = SH(l, m, theta, phi)

    return outs


def _quat_normalize(q, eps=1e-8):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def _quat_rotate(q, v):
    w = q[..., :1]
    qvec = q[..., 1:]
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + w * t + torch.cross(qvec, t, dim=-1)


def herglotz(
    x: torch.Tensor,
    A_real: torch.Tensor,
    A_imag: torch.Tensor,
    sigma_mod: torch.Tensor,
    sigma_arg: torch.Tensor,
    inv_const: torch.Tensor,
    qrot: Optional[torch.Tensor] = None,
):

    if qrot is not None:
        q = _quat_normalize(qrot)
        A_real = _quat_rotate(q, A_real)
        A_imag = _quat_rotate(q, A_imag)

    ax_R = F.linear(x, A_real)  # (..., num_atoms)
    ax_I = F.linear(x, A_imag)

    rho = F.softplus(sigma_mod)
    th = sigma_arg

    c = torch.cos(th)
    s = torch.sin(th)

    r = ax_R * c - ax_I * s
    s_ = ax_R * s + ax_I * c

    exp_term = torch.exp(rho * (r - 1.0))
    cos_term = torch.cos(rho * s_)
    sin_term = torch.sin(rho * s_)

    real_h = exp_term * ((1.0 + 2.0 * rho * r) * cos_term - (2.0 * rho * s_) * sin_term)
    return inv_const * real_h
