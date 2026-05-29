"""
Rectified Flow on 2D toy data (two moons)
==========================================

Reference: Liu et al. 2022, "Flow Straight and Fast" (arxiv 2209.03003)
           Lipman et al. 2022, "Flow Matching" (arxiv 2210.02747)

Pairs with: docs/tutorials/flow_matching_tutorial.md (concept reference).

The three lines that matter:
    x_t = (1-t) * x_0 + t * x_1       # straight interpolation path
    target = x_1 - x_0                 # velocity (constant along the chord)
    x += v_theta(x, t) * dt            # Euler step at inference time

Convention used here: x_0 ~ N(0,I) is noise, x_1 ~ data, t in [0,1] goes
from noise to data. The model regresses (x_1 - x_0) directly.

Run:
    python flow_matching.py
    # CPU ~30s, writes fm_result.png if matplotlib is available.
"""
import math
import torch
import torch.nn as nn

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# --- data --------------------------------------------------------------------

def sample_moons(n: int, noise: float = 0.05, device: str = "cpu") -> torch.Tensor:
    """sklearn-style two-moons data, scaled 2x for visualization."""
    t = torch.rand(n, device=device) * math.pi
    upper = torch.stack([torch.cos(t), torch.sin(t) - 0.5], dim=1)
    flip = torch.rand(n, device=device) > 0.5
    upper[flip] = torch.stack(
        [1 - torch.cos(t[flip]), -torch.sin(t[flip]) + 0.5], dim=1
    )
    return (upper + noise * torch.randn_like(upper)) * 2.0


# --- model -------------------------------------------------------------------

class TimeMLP(nn.Module):
    """Simple MLP velocity field: (x, t) -> v. Concatenates t into the input."""

    def __init__(self, dim_in: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_in),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim_in]
            t: [B]   in [0, 1]
        Returns:
            [B, dim_in]
        """
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))


# --- train -------------------------------------------------------------------

def train(
    model: nn.Module,
    steps: int = 4000,
    bs: int = 1024,
    lr: float = 2e-3,
    device: str = "cpu",
) -> nn.Module:
    """Rectified Flow training: regress velocity (x_1 - x_0)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        x1 = sample_moons(bs, device=device)        # data sample (x_1)
        x0 = torch.randn_like(x1)                   # noise prior (x_0)
        t = torch.rand(bs, device=device)           # uniform t in [0, 1]
        xt = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
        target = x1 - x0                            # <- rectified flow velocity
        pred = model(xt, t)
        loss = (pred - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    return model


# --- sample ------------------------------------------------------------------

@torch.no_grad()
def sample(model: nn.Module, n: int = 2000, nfe: int = 50, device: str = "cpu") -> torch.Tensor:
    """Euler ODE integration: x += v(x, t) * dt, t goes 0 -> 1."""
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / nfe
    for k in range(nfe):
        t = torch.full((n,), k * dt, device=device)
        x = x + model(x, t) * dt
    return x.cpu()


@torch.no_grad()
def sample_trajectory(
    model: nn.Module, n: int = 200, nfe: int = 50, device: str = "cpu"
) -> torch.Tensor:
    """Record sampling trajectory per step, shape [nfe+1, n, 2]."""
    x = torch.randn(n, 2, device=device)
    traj = [x.clone().cpu()]
    dt = 1.0 / nfe
    for k in range(nfe):
        t = torch.full((n,), k * dt, device=device)
        x = x + model(x, t) * dt
        traj.append(x.clone().cpu())
    return torch.stack(traj)                        # [nfe+1, n, 2]


# --- viz ---------------------------------------------------------------------

def plot_results(real, fake, traj, fname: str = "fm_result.png"):
    if not HAS_PLT:
        print(f"[plot] matplotlib not available; skipping {fname}")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(real[:, 0], real[:, 1], s=2, c="C0")
    axes[0].set_title("Real (two moons)")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)

    axes[1].scatter(fake[:, 0], fake[:, 1], s=2, c="C1")
    axes[1].set_title("Generated (RF, 50 NFE)")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)

    nfe = traj.shape[0] - 1
    for i in range(traj.shape[1]):
        axes[2].plot(traj[:, i, 0], traj[:, i, 1], alpha=0.15, lw=0.5, c="grey")
    axes[2].scatter(traj[0, :, 0], traj[0, :, 1], s=5, c="C0", label="x_0 (noise)")
    axes[2].scatter(traj[-1, :, 0], traj[-1, :, 1], s=5, c="C1", label=f"x_1 (after {nfe} steps)")
    axes[2].set_title("Sampling trajectories")
    axes[2].legend()
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    print(f"[plot] saved {fname}")


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model = TimeMLP().to(device)
    print("training RF on two-moons...")
    train(model, steps=4000, bs=1024, device=device)

    real = sample_moons(4000, device=device).cpu()
    fake = sample(model, n=4000, nfe=50, device=device)
    traj = sample_trajectory(model, n=100, nfe=50, device=device)

    print("\n--- sanity ---")
    print(f"real mean  = {real.mean(0).tolist()}, std = {real.std(0).tolist()}")
    print(f"fake mean  = {fake.mean(0).tolist()}, std = {fake.std(0).tolist()}")

    plot_results(real, fake, traj)


if __name__ == "__main__":
    main()
