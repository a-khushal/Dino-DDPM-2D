import argparse
import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class NoisePredictor(nn.Module):
    def __init__(self, x_dim: int = 2, t_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_embed(t)
        h = torch.cat([x, te], dim=1)
        return self.net(h)


def gather_t(vals: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = vals.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def make_beta_schedule(T: int, device: str, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, T, device=device)
    if schedule == "cosine":
        s = 0.008
        t = torch.linspace(0, T, T + 1, device=device, dtype=torch.float32)
        f = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-5, 0.999)
    raise ValueError(f"Unknown schedule: {schedule}")


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for sp, p in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(d).add_(p.data, alpha=1.0 - d)

        for sb, b in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(b)


def load_dino_points(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep="\t")
    pts = df.loc[df["dataset"] == "dino", ["x", "y"]].to_numpy(dtype=np.float32)
    if len(pts) == 0:
        raise ValueError("No rows found where dataset == 'dino'.")
    return pts


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    n_samples: int,
    T: int,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    alpha_bars_prev: torch.Tensor,
    betas: torch.Tensor,
    posterior_variance: torch.Tensor,
    device: str,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(n_samples, 2, device=device)

    for ti in reversed(range(T)):
        t = torch.full((n_samples,), ti, device=device, dtype=torch.long)
        t_scaled = t.float() / T
        eps_theta = model(x, t_scaled)

        alpha_t = alphas[ti]
        alpha_bar_t = alpha_bars[ti]
        beta_t = betas[ti]

        mu = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
        )

        if ti > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt(posterior_variance[ti])
            x = mu + sigma * z
        else:
            x = mu

    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDPM on dino 2D points.")
    parser.add_argument(
        "--data", type=str, default="Datashape.tsv", help="Path to Datashape.tsv"
    )
    parser.add_argument("--steps", type=int, default=8000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--timesteps", type=int, default=300, help="Diffusion timesteps"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden width for MLP")
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Beta schedule",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="EMA decay for sampling model"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of points to generate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out", type=str, default="dino_overlay.png", help="Output plot path"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    real = load_dino_points(args.data)
    mean = real.mean(axis=0, keepdims=True)
    std = real.std(axis=0, keepdims=True) + 1e-8
    x0_np = (real - mean) / std

    x0 = torch.tensor(x0_np, dtype=torch.float32, device=device)
    n_data = x0.shape[0]
    print(f"dino points: {n_data}")

    T = args.timesteps
    betas = make_beta_schedule(T=T, device=device, schedule=args.schedule)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat(
        [torch.ones(1, device=device, dtype=alpha_bars.dtype), alpha_bars[:-1]], dim=0
    )
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_variance = posterior_variance.clamp(min=1e-20)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    model = NoisePredictor(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, decay=args.ema_decay)

    loss_hist = []
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n_data, (args.batch_size,), device=device)
        xb = x0[idx]
        t = torch.randint(0, T, (args.batch_size,), device=device).long()

        eps = torch.randn_like(xb)
        s1 = gather_t(sqrt_alpha_bars, t, xb.shape)
        s2 = gather_t(sqrt_one_minus_alpha_bars, t, xb.shape)
        xt = s1 * xb + s2 * eps

        t_scaled = t.float() / T
        eps_pred = model(xt, t_scaled)
        loss = F.mse_loss(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        ema.update(model)

        loss_hist.append(loss.item())
        if step % 500 == 0:
            print(f"step {step:5d} | avg loss {np.mean(loss_hist[-500:]):.6f}")

    gen_norm = (
        sample_ddpm(
            model=ema.shadow,
            n_samples=args.samples,
            T=T,
            alphas=alphas,
            alpha_bars=alpha_bars,
            alpha_bars_prev=alpha_bars_prev,
            betas=betas,
            posterior_variance=posterior_variance,
            device=device,
        )
        .cpu()
        .numpy()
    )
    gen = gen_norm * std + mean

    plt.figure(figsize=(6, 6))
    plt.scatter(real[:, 0], real[:, 1], s=18, alpha=0.7, label="real dino")
    plt.scatter(gen[:, 0], gen[:, 1], s=10, alpha=0.55, label="generated")
    plt.title("DDPM on 2D Dino Dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved overlay plot: {args.out}")

    plt.figure(figsize=(6, 3))
    plt.plot(loss_hist)
    plt.title("Training Loss")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.tight_layout()
    loss_plot = args.out.rsplit(".", 1)[0] + "_loss.png"
    plt.savefig(loss_plot, dpi=150)
    print(f"saved loss plot: {loss_plot}")


if __name__ == "__main__":
    main()
