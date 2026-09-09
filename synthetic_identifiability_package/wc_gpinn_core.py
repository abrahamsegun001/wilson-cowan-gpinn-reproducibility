#!/usr/bin/env python3
"""Core Wilson-Cowan gPINN components adapted from the real-data training script.

The inverse model intentionally preserves the architecture, residual scaling,
parameter initialization, dynamic loss balancing, and Adam -> L-BFGS schedule
used in train_gpinn_narval_updated.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DecoupledFourierEmbedding(nn.Module):
    def __init__(
        self,
        ff_dim: int = 128,
        scale_xy: float = 25.0,
        scale_t: float = 50.0,
        fourier_seed: int | None = None,
    ) -> None:
        super().__init__()
        if ff_dim % 2 != 0:
            raise ValueError("ff_dim must be even.")

        if fourier_seed is None:
            B_xy = torch.randn(2, ff_dim // 2) * scale_xy
            B_t = torch.randn(1, ff_dim // 2) * scale_t
        else:
            # Use a local generator so a fixed Fourier realization does not
            # alter the global RNG used to initialize the trainable MLP.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(fourier_seed))
            B_xy = torch.randn(2, ff_dim // 2, generator=gen) * scale_xy
            B_t = torch.randn(1, ff_dim // 2, generator=gen) * scale_t

        self.register_buffer("B_xy", B_xy)
        self.register_buffer("B_t", B_t)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        xy = xyt[:, :2]
        t = xyt[:, 2:3]
        proj_xy = xy @ self.B_xy
        proj_t = t @ self.B_t
        proj = torch.cat([proj_xy, proj_t], dim=-1)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class gPINN_WilsonCowan(nn.Module):
    def __init__(
        self,
        ff_dim: int = 128,
        hidden_neurons: int = 256,
        hidden_layers: int = 6,
        d_min: float = 0.1,
        d_max: float = 5.0,
        fourier_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_min = float(d_min)
        self.d_max = float(d_max)

        self.embed_E = DecoupledFourierEmbedding(
            ff_dim=ff_dim,
            scale_xy=25.0,
            scale_t=50.0,
            fourier_seed=fourier_seed,
        )
        embed_dim_E = 2 * ff_dim

        net_e = [nn.Linear(embed_dim_E, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            net_e += [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
        net_e += [nn.Linear(hidden_neurons, 1)]
        self.net_E = nn.Sequential(*net_e)

        # Same parameter initialization as the real-data code.
        self.raw_D = nn.Parameter(torch.tensor(0.0))       # D_E = 2.55 initially
        self.w_EE = nn.Parameter(torch.tensor(2.0))
        self.h = nn.Parameter(torch.tensor(-1.0))

    def forward_E(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net_E(self.embed_E(xyt))

    def forward_D(self, xy: torch.Tensor | None = None) -> torch.Tensor:
        return self.d_min + (self.d_max - self.d_min) * torch.sigmoid(self.raw_D)

    @staticmethod
    def sigmoid_S(u: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(u)


def gpinn_residual(
    model: gPINN_WilsonCowan,
    xyt: torch.Tensor,
    tau: float = 1.00,
    chunk_duration: float = 30.0,
    l_fov: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyt = xyt.clone().requires_grad_(True)
    E = model.forward_E(xyt)
    D_E = model.forward_D()
    ones = torch.ones_like(E)

    g = torch.autograd.grad(E, xyt, ones, create_graph=True)[0]
    dE_dt = g[:, 2:3]
    dE_dx = g[:, 0:1]
    dE_dy = g[:, 1:2]

    d2E_dx2 = torch.autograd.grad(
        dE_dx, xyt, ones, create_graph=True, retain_graph=True
    )[0][:, 0:1]
    d2E_dy2 = torch.autograd.grad(
        dE_dy, xyt, ones, create_graph=True
    )[0][:, 1:2]

    lap_E = 4.0 * (d2E_dx2 + d2E_dy2) / (l_fov ** 2)
    rhs = model.sigmoid_S(model.w_EE * E + D_E * lap_E + model.h)
    f = 2.0 * (tau / chunk_duration) * dE_dt - (-E + rhs)

    df_dx = torch.autograd.grad(
        f, xyt, torch.ones_like(f), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    df_dy = torch.autograd.grad(
        f, xyt, torch.ones_like(f), create_graph=True
    )[0][:, 1:2]
    return f, df_dx, df_dy


def compute_loss(
    model: gPINN_WilsonCowan,
    Xt: torch.Tensor,
    Et: torch.Tensor,
    Xc: torch.Tensor,
    lam: float,
    mu_grad: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    E_pred = model.forward_E(Xt)
    data_loss = torch.mean((E_pred - Et) ** 2)
    f, df_dx, df_dy = gpinn_residual(model, Xc)
    physics_loss = torch.mean(f ** 2)
    gradient_loss = torch.mean(df_dx ** 2) + torch.mean(df_dy ** 2)
    total = data_loss + lam * physics_loss + lam * mu_grad * gradient_loss
    return total, data_loss, physics_loss, gradient_loss


def compute_balanced_lambda(
    model: gPINN_WilsonCowan,
    Xt: torch.Tensor,
    Et: torch.Tensor,
    Xc: torch.Tensor,
    current_lambda: float,
) -> float:
    model.zero_grad(set_to_none=True)
    data_loss = torch.mean((model.forward_E(Xt) - Et) ** 2)
    data_loss.backward()
    gd = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    grad_data = float(np.mean(gd)) if gd else 0.0

    model.zero_grad(set_to_none=True)
    f, _, _ = gpinn_residual(model, Xc)
    torch.mean(f ** 2).backward()
    gp = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    grad_phys = float(np.mean(gp)) if gp else 0.0

    model.zero_grad(set_to_none=True)
    if grad_phys < 1e-10:
        return float(current_lambda)
    return float(np.clip(grad_data / grad_phys, 0.05, 1.0))


def evaluate_metrics(
    model: gPINN_WilsonCowan,
    Xt_val: torch.Tensor,
    Et_val: torch.Tensor,
) -> Tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        pred = model.forward_E(Xt_val).cpu().numpy().ravel()
        true = Et_val.cpu().numpy().ravel()
    mse = float(np.mean((pred - true) ** 2))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    r = float(pearsonr(pred, true)[0]) if np.std(pred) > 0 and np.std(true) > 0 else float("nan")
    model.train()
    return mse, r2, r


@dataclass
class TrainResult:
    history: Dict[str, list]
    val_mse: float
    val_r2: float
    val_r: float
    final_lambda: float


def train_single_chunk(
    model: gPINN_WilsonCowan,
    Xt_tr: torch.Tensor,
    Et_tr: torch.Tensor,
    Xt_va: torch.Tensor,
    Et_va: torch.Tensor,
    Xc: torch.Tensor,
    lr: float = 1e-3,
    adam_epochs: int = 5000,
    lbfgs_steps: int = 500,
    lambda_phys: float = 0.01,
    w_bounds: tuple[float, float] = (0.5, 6.0),
    mu_grad: float = 1e-5,
    log_every: int = 50,
) -> TrainResult:
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr)
    lam = float(lambda_phys)
    history = {
        "total": [], "data": [], "physics": [], "gradient": [], "lambda": [],
        "w_EE": [], "h": [], "D_E": [], "raw_D": [], "step": []
    }

    model.train()
    for step in range(adam_epochs):
        if step % 200 == 0 and step > 0:
            lam = compute_balanced_lambda(model, Xt_tr, Et_tr, Xc, lam)

        optimizer_adam.zero_grad(set_to_none=True)
        loss, l_data, l_phys, l_grad = compute_loss(
            model, Xt_tr, Et_tr, Xc, lam, mu_grad=mu_grad
        )
        loss.backward()
        optimizer_adam.step()
        with torch.no_grad():
            model.w_EE.clamp_(*w_bounds)

        if step % log_every == 0 or step == adam_epochs - 1:
            history["step"].append(int(step))
            history["total"].append(float(loss.item()))
            history["data"].append(float(l_data.item()))
            history["physics"].append(float(l_phys.item()))
            history["gradient"].append(float(l_grad.item()))
            history["lambda"].append(float(lam))
            history["w_EE"].append(float(model.w_EE.item()))
            history["h"].append(float(model.h.item()))
            history["D_E"].append(float(model.forward_D().item()))
            history["raw_D"].append(float(model.raw_D.item()))

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_steps,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer_lbfgs.zero_grad(set_to_none=True)
        total, _, _, _ = compute_loss(model, Xt_tr, Et_tr, Xc, lam, mu_grad=mu_grad)
        total.backward()
        # Keep the real-data hard clamp behavior. A post-step clamp is also
        # applied below because L-BFGS can update parameters after closure calls.
        with torch.no_grad():
            model.w_EE.clamp_(*w_bounds)
        return total

    optimizer_lbfgs.step(closure)
    with torch.no_grad():
        model.w_EE.clamp_(*w_bounds)

    val_mse, val_r2, val_r = evaluate_metrics(model, Xt_va, Et_va)
    return TrainResult(history, val_mse, val_r2, val_r, lam)
