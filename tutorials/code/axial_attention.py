"""
Axial Attention - minimal runnable implementation
=================================================

Reference: Ho et al. 2019, "Axial Attention in Multidimensional Transformers"
           (arxiv 1912.12180)

Core idea:
    Vanilla 2D self-attention has O((HW)^2 * d) cost which is prohibitive
    for high-resolution feature maps.

    Axial attention factorizes it: one attention pass along the H axis,
    another along the W axis.
        H-axial: fold W into batch; each column attends within its H tokens
        W-axial: fold H into batch; each row attends within its W tokens

    Complexity drops from O((HW)^2 d) to O(HW(H+W) d), saving HW/(H+W) factor.

    Stacking one H-axial then one W-axial gives an attention path between any
    two positions (h1, w1) -> (h2, w2), but it's a factorized two-step
    propagation, NOT equivalent to a single dense 2D attention.

Run:
    python axial_attention.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialAttention(nn.Module):
    """1D axial attention over a specific spatial dim.

    Args:
        channels: input/output channel count C
        d_k:      attention head dim
        axis:     'H' or 'W' - which spatial axis to attend along
    """

    def __init__(self, channels: int, d_k: int, axis: str):
        super().__init__()
        assert axis in ("H", "W")
        self.axis = axis
        self.d_k = d_k
        self.W_q = nn.Linear(channels, d_k, bias=False)
        self.W_k = nn.Linear(channels, d_k, bias=False)
        self.W_v = nn.Linear(channels, d_k, bias=False)
        self.out = nn.Linear(d_k, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape

        if self.axis == "H":
            # Fold W into batch: [B, C, H, W] -> [B*W, H, C]
            x_in = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        else:  # axis == 'W'
            # Fold H into batch: [B, C, H, W] -> [B*H, W, C]
            x_in = x.permute(0, 2, 3, 1).reshape(B * H, W, C)

        Q = self.W_q(x_in)                          # [B*?, seq, d_k]
        K = self.W_k(x_in)
        V = self.W_v(x_in)

        score = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(score, dim=-1)
        out = attn @ V                              # [B*?, seq, d_k]
        out = self.out(out)                         # [B*?, seq, C]

        # Restore spatial shape.
        if self.axis == "H":
            # [B*W, H, C] -> [B, W, H, C] -> [B, C, H, W]
            out = out.reshape(B, W, H, C).permute(0, 3, 2, 1)
        else:
            # [B*H, W, C] -> [B, H, W, C] -> [B, C, H, W]
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class AxialBlock(nn.Module):
    """H-axial + W-axial stacked.

    Two layers in series create an attention path between any (h1, w1)
    and (h2, w2) -- the H-axial pass moves information along column w1 to
    (h2, w1), then the W-axial pass moves it to (h2, w2). But this is a
    factorized two-step propagation, NOT equivalent to one dense 2D attention.
    The combined "weights" are a product of two 1D softmaxes; expressivity is
    strictly weaker than full 2D attention. The win is O(HW(H+W)d) vs
    O(H^2 W^2 d).
    """

    def __init__(self, channels: int, d_k: int):
        super().__init__()
        self.h_attn = AxialAttention(channels, d_k, axis="H")
        self.w_attn = AxialAttention(channels, d_k, axis="W")
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        x = x + self.h_attn(self.norm1(x))          # residual
        x = x + self.w_attn(self.norm2(x))
        return x


# --- complexity comparison ----------------------------------------------------

def complexity_vanilla(H: int, W: int, d: int) -> int:
    """Vanilla 2D self-attention FLOPs ~ 2 * (HW)^2 * d  (QK^T + attn @ V)."""
    return 2 * (H * W) ** 2 * d


def complexity_axial(H: int, W: int, d: int) -> int:
    """Axial: H attention W times + W attention H times, each O(seq^2 d)."""
    return 2 * W * H * H * d + 2 * H * W * W * d


def print_complexity_table():
    print(f"{'H':>4} {'W':>4} | {'vanilla':>14} | {'axial':>14} | {'speedup':>8}")
    print("-" * 60)
    for H, W in [(8, 8), (32, 32), (64, 64), (128, 128), (256, 256)]:
        d = 64
        v = complexity_vanilla(H, W, d)
        a = complexity_axial(H, W, d)
        print(f"{H:>4} {W:>4} | {v:>14,} | {a:>14,} | {v / a:>8.1f}x")


# --- sanity check -------------------------------------------------------------

def shape_check():
    B, C, H, W = 2, 32, 16, 16
    x = torch.randn(B, C, H, W)

    h_only = AxialAttention(C, d_k=32, axis="H")
    w_only = AxialAttention(C, d_k=32, axis="W")
    block = AxialBlock(C, d_k=32)

    y_h = h_only(x)
    y_w = w_only(x)
    y_b = block(x)

    print(f"[shape] x        : {tuple(x.shape)}")
    print(f"[shape] H-axial  : {tuple(y_h.shape)}")
    print(f"[shape] W-axial  : {tuple(y_w.shape)}")
    print(f"[shape] H+W block: {tuple(y_b.shape)}")
    assert y_h.shape == y_w.shape == y_b.shape == x.shape


def receptive_field_check():
    """Verify a single H-axial layer keeps columns isolated.

    Perturbing one pixel at (row=0, col=2) should only change outputs in
    column 2 (across all rows), leaving every other column untouched.
    """
    torch.manual_seed(0)
    B, C, H, W = 1, 4, 6, 6
    x = torch.randn(B, C, H, W)

    layer = AxialAttention(C, d_k=8, axis="H")
    y_orig = layer(x)

    # Perturb one pixel in column 2.
    x_perturbed = x.clone()
    x_perturbed[0, :, 0, 2] += 1.0
    y_perturbed = layer(x_perturbed)

    diff = (y_orig - y_perturbed).abs().sum(dim=1)  # [B, H, W]
    print(f"[recep] diff per spatial position after perturbing (row=0, col=2):")
    print(diff[0])
    col2_sum = diff[0, :, 2].sum().item()
    other_sum = (diff[0].sum() - diff[0, :, 2].sum()).item()
    print(f"[recep] col=2 total diff = {col2_sum:.4f}  (should be > 0)")
    print(f"[recep] other cols diff  = {other_sum:.6f}  (should be 0)")
    assert col2_sum > 1e-4 and other_sum < 1e-6
    print("[recep] PASS - H-axial isolates columns as expected")


if __name__ == "__main__":
    print("== Axial Attention ==\n")
    print("--- complexity comparison (FLOPs, d=64) ---")
    print_complexity_table()
    print("\n--- shape check ---")
    shape_check()
    print("\n--- receptive field check ---")
    receptive_field_check()
    print("\nAll checks passed.")
