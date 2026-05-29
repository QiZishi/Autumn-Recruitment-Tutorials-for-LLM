"""
Multi-Head Attention - minimal runnable implementation
======================================================

Educational PyTorch reference for Multi-Head Self-Attention.
Standalone script: runs three sanity checks on CPU in a few seconds.

Pairs with: docs/tutorials/attention_tutorial.md (concept reference).

Key shapes:
    Input  x: [B, N, C]
    qkv:      [B, N, 3*C] -> reshape -> [3, B, H, N, d_k]
    score:    [B, H, N, N]
    Output:   [B, N, C]

Run:
    python mha.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention.

    - Single fused Linear for Q/K/V (kernel-fusion friendly).
    - Optional additive attention mask (float, -inf at masked positions).
    - No KV-cache; meant for educational clarity, not inference speed.
    """

    def __init__(self, embed_dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # One Linear produces Q, K, V at once.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        # Output projection bias follows qkv_bias so the sanity check below can
        # align cleanly with torch.nn.MultiheadAttention (which has no
        # out_proj.bias when bias=False).
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        Args:
            x: [B, N, C]
            attn_mask: [N, N] additive mask (float, -inf at masked positions)
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape

        # qkv: [B, N, 3*C] -> [B, N, 3, H, d_k] -> [3, B, H, N, d_k]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # each: [B, H, N, d_k]

        # score: [B, H, N, N]
        score = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            score = score + attn_mask                # additive mask, broadcasts
        attn = F.softmax(score, dim=-1)

        # out: [B, H, N, d_k] -> [B, N, H, d_k] -> [B, N, C]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


def make_causal_mask(n: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Causal mask of shape [N, N]: upper triangle filled with -inf."""
    mask = torch.zeros(n, n, device=device)
    mask = mask.masked_fill(
        torch.triu(torch.ones(n, n, device=device), diagonal=1).bool(),
        float("-inf"),
    )
    return mask


def sanity_check():
    """Verify alignment with torch.nn.MultiheadAttention (no mask, no bias)."""
    torch.manual_seed(0)
    B, N, C, H = 2, 17, 64, 4
    x = torch.randn(B, N, C)

    mine = MultiHeadAttention(C, H, qkv_bias=False)
    # bias=False so builtin.in_proj_bias / out_proj.bias are both None.
    builtin = nn.MultiheadAttention(C, H, bias=False, batch_first=True)

    # Copy our qkv + proj weights into the builtin (in_proj_weight = [3*C, C]).
    with torch.no_grad():
        builtin.in_proj_weight.copy_(mine.qkv.weight)
        builtin.out_proj.weight.copy_(mine.proj.weight)
        # Note: bias=False -> builtin.out_proj.bias is None, and mine.proj also
        # has no bias because we passed qkv_bias=False (see __init__).

    y_mine = mine(x)
    y_builtin, _ = builtin(x, x, x, need_weights=False)
    diff = (y_mine - y_builtin).abs().max().item()
    print(f"[sanity] max diff vs nn.MultiheadAttention = {diff:.2e}")
    assert diff < 1e-5, f"mismatch! diff={diff}"
    print("[sanity] PASS")


def shape_check():
    """Common transformer shape: B=2, N=1024 (32x32 tokens), C=768, H=12, d_k=64."""
    B, N, C, H = 2, 1024, 768, 12
    x = torch.randn(B, N, C)
    mha = MultiHeadAttention(C, H)
    y = mha(x)
    print(f"[shape] input  x: {tuple(x.shape)}")
    print(f"[shape] output y: {tuple(y.shape)}")
    assert y.shape == x.shape


def causal_check():
    """Causal mask sanity check for autoregressive transformers."""
    B, N, C, H = 1, 8, 32, 4
    x = torch.randn(B, N, C)
    mha = MultiHeadAttention(C, H)
    mask = make_causal_mask(N)
    y = mha(x, attn_mask=mask)
    print(f"[causal] applied causal mask, output shape {tuple(y.shape)}")


if __name__ == "__main__":
    print("== Multi-Head Attention sanity ==")
    sanity_check()
    shape_check()
    causal_check()
    print("\nAll checks passed.")
