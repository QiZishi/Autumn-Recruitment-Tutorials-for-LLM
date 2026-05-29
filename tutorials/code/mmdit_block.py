"""
MMDiT block - minimal runnable implementation
=============================================

Educational toy version of one double-stream MMDiT block, a building block
commonly used by modern open MMDiT-family rectified-flow text-to-image
transformers.

Pairs with: docs/tutorials/image_generation_systems_tutorial.md (concept).

Architecture:
    - Two streams (text + image) with independent Q/K/V projections
    - Concatenate along the seq dim and run a single joint attention
    - Split outputs back and run independent FFNs per stream
    - AdaLN-Zero gating: timestep -> 6-way (shift, scale, gate) per sublayer

Hidden sizes here are deliberately small so the whole script runs on CPU.

Run:
    python mmdit_block.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Timestep embedding (sinusoidal, transformer-style)
# ============================================================================

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal time embedding.

    Args:
        t: [B] timestep (any positive scale, commonly 0-1000)
        dim: embedding dim
    Returns:
        [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:                                # pad to dim
        emb = F.pad(emb, (0, 1))
    return emb


# ============================================================================
# AdaLN-Zero modulation (a common diffusion-transformer pattern)
# ============================================================================

class AdaLNZero(nn.Module):
    """Project timestep embedding to 6 modulation chunks:

        (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)

    Zero-init so each block starts as identity, a stabilization trick widely
    used by modern diffusion-transformer architectures.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim)
        # zero init - gates start at 0 => block is initially identity
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, temb: torch.Tensor) -> tuple:
        """Returns 6 tensors of shape [B, hidden_dim]."""
        out = self.linear(self.silu(temb))
        return out.chunk(6, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x <- (1 + scale) * x + shift.  shift/scale broadcast over the seq dim."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================================
# Joint Attention (the heart of MMDiT)
# ============================================================================

class JointAttention(nn.Module):
    """Independent Q/K/V proj per stream; concat along seq and attend once.

    Each stream (text, image) has its own input/output projection. We then
    concatenate Q/K/V along the seq dim, run a single SDPA, and split the
    output back into the two streams.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # text stream
        self.txt_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.txt_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.txt_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.txt_out = nn.Linear(hidden_dim, hidden_dim)

        # image stream
        self.img_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.img_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.img_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.img_out = nn.Linear(hidden_dim, hidden_dim)

        # QK normalization (RMS or LN) before attention, a common stability
        # trick in modern diffusion-transformer variants.
        norm_cls = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.q_norm_txt = norm_cls(self.head_dim)
        self.k_norm_txt = norm_cls(self.head_dim)
        self.q_norm_img = norm_cls(self.head_dim)
        self.k_norm_img = norm_cls(self.head_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, C] -> [B, H, N, d_k]"""
        B, N, C = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, N, d_k] -> [B, N, C]"""
        B, H, N, d = x.shape
        return x.transpose(1, 2).reshape(B, N, H * d)

    def forward(self, txt: torch.Tensor, img: torch.Tensor) -> tuple:
        """
        Args:
            txt: [B, L_txt, C]
            img: [B, L_img, C]
        Returns:
            (txt_out, img_out) - each with the original shape.
        """
        # Per-stream Q/K/V projections.
        q_t = self._split_heads(self.txt_q(txt))
        k_t = self._split_heads(self.txt_k(txt))
        v_t = self._split_heads(self.txt_v(txt))

        q_i = self._split_heads(self.img_q(img))
        k_i = self._split_heads(self.img_k(img))
        v_i = self._split_heads(self.img_v(img))

        # QK normalization before attention (positional encoding such as RoPE
        # would typically be applied here in a full implementation; omitted
        # for simplicity).
        q_t = self.q_norm_txt(q_t)
        k_t = self.k_norm_txt(k_t)
        q_i = self.q_norm_img(q_i)
        k_i = self.k_norm_img(k_i)

        # Concatenate along seq (text first by convention).
        L_txt = txt.shape[1]
        q = torch.cat([q_t, q_i], dim=2)            # [B, H, L_txt + L_img, d_k]
        k = torch.cat([k_t, k_i], dim=2)
        v = torch.cat([v_t, v_i], dim=2)

        # Single scaled-dot-product attention call (PyTorch will dispatch to
        # a memory-efficient attention kernel when available).
        out = F.scaled_dot_product_attention(q, k, v)  # [B, H, L_total, d_k]
        out = self._merge_heads(out)                  # [B, L_total, C]

        # Split back into text / image.
        txt_out = self.txt_out(out[:, :L_txt])
        img_out = self.img_out(out[:, L_txt:])
        return txt_out, img_out


# ============================================================================
# FFN (per-stream, GELU)
# ============================================================================

class FFN(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ============================================================================
# MMDiT Block - one complete double-stream layer
# ============================================================================

class MMDiTBlock(nn.Module):
    """One complete double-stream MMDiT block:

        (LN -> modulate -> JointAttn -> gate * residual)
        (LN -> modulate -> FFN -> gate * residual)   -- per stream

    Reference: standard double-stream MMDiT block as used by modern open
    rectified-flow text-to-image transformers.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm_attn_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_attn_img = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_mlp_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_mlp_img = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.adaln_txt = AdaLNZero(hidden_dim)
        self.adaln_img = AdaLNZero(hidden_dim)

        self.attn = JointAttention(hidden_dim, num_heads)
        self.ffn_txt = FFN(hidden_dim)
        self.ffn_img = FFN(hidden_dim)

    def forward(
        self, txt: torch.Tensor, img: torch.Tensor, temb: torch.Tensor
    ) -> tuple:
        """
        Args:
            txt:  [B, L_txt, C]
            img:  [B, L_img, C]
            temb: [B, C]   - timestep embedding
        """
        # AdaLN params: (shift, scale, gate) x (attn, mlp), independent per stream.
        sa_t, sc_t, ga_t, sm_t, scm_t, gm_t = self.adaln_txt(temb)
        sa_i, sc_i, ga_i, sm_i, scm_i, gm_i = self.adaln_img(temb)

        # === Joint Attention sublayer ===
        t_norm = modulate(self.norm_attn_txt(txt), sa_t, sc_t)
        i_norm = modulate(self.norm_attn_img(img), sa_i, sc_i)
        t_attn, i_attn = self.attn(t_norm, i_norm)
        txt = txt + ga_t.unsqueeze(1) * t_attn
        img = img + ga_i.unsqueeze(1) * i_attn

        # === FFN sublayer (per-stream) ===
        t_mlp = self.ffn_txt(modulate(self.norm_mlp_txt(txt), sm_t, scm_t))
        i_mlp = self.ffn_img(modulate(self.norm_mlp_img(img), sm_i, scm_i))
        txt = txt + gm_t.unsqueeze(1) * t_mlp
        img = img + gm_i.unsqueeze(1) * i_mlp

        # FP16 overflow guard, useful when running large models in fp16/bf16.
        txt = torch.clamp(txt, -65504, 65504)
        img = torch.clamp(img, -65504, 65504)
        return txt, img


# ============================================================================
# Demo
# ============================================================================

def demo():
    torch.manual_seed(0)
    B = 2
    L_txt, L_img = 128, 256                         # example sequence lengths
    hidden, heads = 256, 8                          # toy hidden, far smaller
                                                    # than production MMDiT
    txt = torch.randn(B, L_txt, hidden)
    img = torch.randn(B, L_img, hidden)
    t = torch.rand(B) * 1000                        # timestep 0-1000

    temb = timestep_embedding(t, hidden)
    block = MMDiTBlock(hidden, heads)

    print(f"input  txt:  {tuple(txt.shape)}")
    print(f"input  img:  {tuple(img.shape)}")
    print(f"input  temb: {tuple(temb.shape)}")

    txt_out, img_out = block(txt, img, temb)
    print(f"output txt:  {tuple(txt_out.shape)}")
    print(f"output img:  {tuple(img_out.shape)}")
    assert txt_out.shape == txt.shape and img_out.shape == img.shape

    # AdaLN-Zero starts at 0 -> block output should equal input (identity check).
    diff_txt = (txt_out - txt).abs().max().item()
    diff_img = (img_out - img).abs().max().item()
    print(f"\n[init-identity] max |txt_out - txt| = {diff_txt:.2e}")
    print(f"[init-identity] max |img_out - img| = {diff_img:.2e}")
    print("(both should be ~0: AdaLN-Zero gate init=0 => block starts identity)")
    assert diff_txt < 1e-5 and diff_img < 1e-5

    total = sum(p.numel() for p in block.parameters())
    print(f"\n[params] single block: {total/1e6:.2f}M  (toy hidden={hidden})")


if __name__ == "__main__":
    print("== MMDiT Block (double-stream, AdaLN-Zero, joint attention) ==\n")
    demo()
