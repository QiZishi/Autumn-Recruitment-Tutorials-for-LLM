"""
Toy MMDiT text-to-image pipeline (end-to-end skeleton)
======================================================

Educational reference that wires together a minimal text encoder, a small
VAE, an MMDiT transformer, a Flow-Matching Euler scheduler, and
norm-preserving classifier-free guidance into a single end-to-end
generation loop.

Component sizes are intentionally small (a few M params total) so the whole
pipeline runs on CPU in seconds. This is NOT a real text-to-image model --
the random-init network produces noise. The point is to verify shapes,
trace data flow, and demonstrate how the parts fit together.

Pairs with:
    docs/tutorials/image_generation_systems_tutorial.md
    docs/tutorials/flow_matching_tutorial.md

Components mirrored (toy versions):
    1. Frozen text encoder           -> ToyTextEncoder
    2. Latent-space autoencoder      -> ToyVAE  (8x spatial, 16 ch)
    3. Double-stream MMDiT           -> ToyMMDiT (reuses MMDiTBlock from
                                        mmdit_block.py)
    4. Flow-Matching Euler scheduler -> FlowMatchEulerScheduler
    5. Norm-preserving CFG           -> true_cfg

Run:
    python toy_mmdit_t2i_pipeline.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pull MMDiTBlock + timestep_embedding from sibling script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmdit_block import MMDiTBlock, timestep_embedding


# ============================================================================
# Toy Text Encoder (stand-in for any frozen LM-style encoder)
# ============================================================================

class ToyTextEncoder(nn.Module):
    """Random token embedding + one self-attn layer.

    Real-world MMDiT pipelines commonly use a frozen pretrained text encoder,
    much larger than this toy module.
    """

    def __init__(self, vocab_size: int = 32000, hidden: int = 256, max_len: int = 128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.layer = nn.TransformerEncoderLayer(
            hidden, nhead=8, dim_feedforward=4 * hidden, batch_first=True
        )

    @torch.no_grad()
    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L_txt] long
        Returns:
            [B, L_txt, hidden]
        """
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.layer(x)


# ============================================================================
# Toy VAE (8x spatial compression, 16 latent channels)
# ============================================================================

class ToyVAE(nn.Module):
    """Conv encoder/decoder, 8x downsample, 16 latent channels.

    Common in modern image-diffusion pipelines: input pixels are compressed
    to a small latent grid before the transformer runs, saving compute and
    memory. 8x spatial compression combined with a modest latent channel
    count is a widely-used pattern; specific configurations vary across
    open releases.
    """

    def __init__(self, latent_ch: int = 16):
        super().__init__()
        # encoder: H,W -> H/8, W/8
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # /2
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # /4
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /8
            nn.SiLU(),
            nn.Conv2d(128, latent_ch, 1),
        )
        # decoder: H/8, W/8 -> H, W
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, 128, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, latent_ch, H/8, W/8]"""
        return self.enc(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, latent_ch, H/8, W/8] -> [B, 3, H, W]"""
        return self.dec(z)


# ============================================================================
# Patch Packing (rearrange 2x2 latent patches into tokens)
# ============================================================================

def pack_latents(z: torch.Tensor, patch: int = 2) -> torch.Tensor:
    """
    [B, C, H, W] -> [B, H/p * W/p, C*p*p]

    Rearrange each patch x patch block into the channel dimension, turning
    a spatial latent into a sequence of tokens. For latent_ch=16 and
    patch=2 this yields token feature dim = 64.
    """
    B, C, H, W = z.shape
    assert H % patch == 0 and W % patch == 0
    z = z.reshape(B, C, H // patch, patch, W // patch, patch)
    z = z.permute(0, 2, 4, 1, 3, 5).contiguous()
    return z.reshape(B, (H // patch) * (W // patch), C * patch * patch)


def unpack_latents(x: torch.Tensor, h_patches: int, w_patches: int, patch: int = 2) -> torch.Tensor:
    """Inverse of pack_latents: [B, N, C*p*p] -> [B, C, H, W]"""
    B, N, Cpp = x.shape
    C = Cpp // (patch * patch)
    x = x.reshape(B, h_patches, w_patches, C, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(B, C, h_patches * patch, w_patches * patch)


# ============================================================================
# MMDiT Transformer (toy size, reusing MMDiTBlock)
# ============================================================================

class ToyMMDiT(nn.Module):
    """Tiny MMDiT for shape verification. Production systems use much larger
    hidden dims and many more layers. This toy uses 4 layers and hidden=256
    so the whole thing runs on CPU in seconds."""

    def __init__(
        self,
        latent_ch: int = 16,
        patch: int = 2,
        text_hidden: int = 256,
        hidden: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.patch = patch
        self.hidden = hidden
        in_dim = latent_ch * patch * patch              # 16 * 2 * 2 = 64

        self.img_in = nn.Linear(in_dim, hidden)
        self.txt_in = nn.Linear(text_hidden, hidden)    # adapter to MMDiT dim

        self.blocks = nn.ModuleList(
            [MMDiTBlock(hidden, num_heads) for _ in range(num_layers)]
        )

        # final projection back to packed-latent dim
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.adaln_final = nn.Linear(hidden, 2 * hidden)
        self.proj_out = nn.Linear(hidden, in_dim)
        nn.init.zeros_(self.adaln_final.weight)
        nn.init.zeros_(self.adaln_final.bias)

    def forward(
        self,
        latent: torch.Tensor,                           # [B, C_lat, H_lat, W_lat]
        text_emb: torch.Tensor,                         # [B, L_txt, C_text]
        t: torch.Tensor,                                # [B]
    ) -> torch.Tensor:
        """Returns velocity prediction in latent space, same shape as latent."""
        B, C_lat, H_lat, W_lat = latent.shape

        # patch packing
        img_tokens = pack_latents(latent, patch=self.patch)  # [B, N, C*p*p]
        img_tokens = self.img_in(img_tokens)                  # [B, N, hidden]

        # text adapter
        txt_tokens = self.txt_in(text_emb)                    # [B, L_txt, hidden]

        # timestep emb
        temb = timestep_embedding(t, self.hidden)             # [B, hidden]

        # MMDiT blocks
        txt = txt_tokens
        img = img_tokens
        for block in self.blocks:
            txt, img = block(txt, img, temb)

        # final modulation + proj
        shift, scale = self.adaln_final(F.silu(temb)).chunk(2, dim=-1)
        img = self.final_norm(img) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        img = self.proj_out(img)                              # [B, N, C*p*p]

        # unpack
        h_p, w_p = H_lat // self.patch, W_lat // self.patch
        return unpack_latents(img, h_p, w_p, patch=self.patch)


# ============================================================================
# FlowMatchEuler scheduler (simplified)
# ============================================================================

class FlowMatchEulerScheduler:
    """Minimal Flow-Matching Euler scheduler.

        x_{t - dt} = x_t + v(x_t, t) * dt          (dt is negative since
                                                    sigma decreases over time)

    Convention: t=1 is pure noise, t=0 is data. Sigma schedule is uniform on
    [0, 1]; production systems often apply resolution-dependent shifts to
    bias sampling toward high-noise regions.
    """

    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps
        # linear sigma schedule from 1 to 0
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        self.timesteps = self.sigmas[:-1] * 1000        # 0-1000 scale

    def step(self, model_out: torch.Tensor, i: int, x: torch.Tensor) -> torch.Tensor:
        """Euler step: x <- x + v * dt"""
        dt = (self.sigmas[i + 1] - self.sigmas[i]).item()  # negative
        return x + model_out * dt


# ============================================================================
# True CFG (norm-preserving)
# ============================================================================

def true_cfg(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    scale: float = 4.0,
) -> torch.Tensor:
    """Norm-preserving classifier-free guidance.

    Standard CFG combines cond + scale * (cond - uncond) but the norm of the
    result can blow up with large scale, producing over-saturated samples.
    The "true CFG" trick (used by several modern image-diffusion pipelines)
    rescales the combined velocity back to the magnitude of the conditional
    prediction, keeping direction but capping magnitude.
    """
    comb = uncond_pred + scale * (cond_pred - uncond_pred)
    cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
    comb_norm = torch.norm(comb, dim=-1, keepdim=True) + 1e-8
    return comb * (cond_norm / comb_norm)


# ============================================================================
# End-to-end pipeline
# ============================================================================

@torch.no_grad()
def generate(
    prompt_ids: torch.Tensor,                           # [B, L_txt]
    neg_prompt_ids: torch.Tensor,                       # [B, L_txt]
    text_encoder: ToyTextEncoder,
    vae: ToyVAE,
    transformer: ToyMMDiT,
    H: int = 64,                                        # pixel res
    W: int = 64,
    num_steps: int = 50,
    cfg_scale: float = 4.0,
    seed: int = 0,
) -> torch.Tensor:
    """Full toy generation pipeline.

        1. encode prompts (cond + uncond)
        2. sample initial latent noise
        3. for each step: cond_pred + uncond_pred -> CFG -> Euler step
        4. VAE decode
    """
    device = prompt_ids.device
    B = prompt_ids.shape[0]
    torch.manual_seed(seed)

    # 1. text encoding
    txt_cond = text_encoder.encode(prompt_ids)          # [B, L_txt, C_text]
    txt_uncond = text_encoder.encode(neg_prompt_ids)

    # 2. initial noise in latent space (8x downsample)
    H_lat, W_lat = H // 8, W // 8
    latent_ch = 16
    x = torch.randn(B, latent_ch, H_lat, W_lat, device=device)

    # 3. denoising loop
    scheduler = FlowMatchEulerScheduler(num_steps)
    for i, t in enumerate(scheduler.timesteps):
        t_batch = t.expand(B).to(device)
        cond_pred = transformer(x, txt_cond, t_batch)
        uncond_pred = transformer(x, txt_uncond, t_batch)

        # True CFG (norm-preserving). Permute to token layout [B, H*W, C] so
        # the norm is taken over the channel dim, then reshape back.
        B_, C_, H_, W_ = cond_pred.shape
        cond_tok = cond_pred.permute(0, 2, 3, 1).reshape(B_, H_ * W_, C_)
        uncond_tok = uncond_pred.permute(0, 2, 3, 1).reshape(B_, H_ * W_, C_)
        guided_tok = true_cfg(cond_tok, uncond_tok, cfg_scale)
        guided = guided_tok.reshape(B_, H_, W_, C_).permute(0, 3, 1, 2)

        x = scheduler.step(guided, i, x)

    # 4. VAE decode
    image = vae.decode(x)                               # [B, 3, H, W]
    return torch.clamp(image, -1, 1)


# ============================================================================
# Demo
# ============================================================================

def demo():
    print("== Toy MMDiT text-to-image pipeline ==\n")
    torch.manual_seed(0)
    device = "cpu"

    text_encoder = ToyTextEncoder(vocab_size=32000, hidden=256, max_len=128).to(device)
    vae = ToyVAE(latent_ch=16).to(device)
    transformer = ToyMMDiT(
        latent_ch=16, patch=2, text_hidden=256,
        hidden=256, num_layers=4, num_heads=8,
    ).to(device)

    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"[init] toy MMDiT params = {n_params/1e6:.2f}M\n")

    # dummy prompts
    B = 2
    prompt_ids = torch.randint(0, 32000, (B, 32))
    neg_prompt_ids = torch.zeros(B, 32, dtype=torch.long)

    # Single-step shape verify
    print("--- single-step shape verify ---")
    H_pix, W_pix = 64, 64
    latent = torch.randn(B, 16, H_pix // 8, W_pix // 8)
    txt_emb = text_encoder.encode(prompt_ids)
    t = torch.rand(B) * 1000
    print(f"  latent:  {tuple(latent.shape)}")
    print(f"  txt_emb: {tuple(txt_emb.shape)}")
    print(f"  t:       {tuple(t.shape)}")
    out = transformer(latent, txt_emb, t)
    print(f"  velocity output: {tuple(out.shape)}")
    assert out.shape == latent.shape

    # Full pipeline (10 steps quick demo)
    print("\n--- full pipeline (10 steps demo) ---")
    image = generate(
        prompt_ids, neg_prompt_ids,
        text_encoder, vae, transformer,
        H=64, W=64, num_steps=10, cfg_scale=4.0,
    )
    print(f"  generated image: {tuple(image.shape)} (expected [B, 3, 64, 64])")
    assert image.shape == (B, 3, 64, 64)

    # CFG sanity: scale=1.0 should be equivalent to no CFG
    print("\n--- CFG sanity check ---")
    cond = torch.randn(2, 4, 8)
    uncond = torch.randn(2, 4, 8)
    out_s1 = true_cfg(cond, uncond, scale=1.0)
    diff = (out_s1.flatten() - cond.flatten()).abs().max().item()
    print(f"  CFG(scale=1.0) vs cond: max diff = {diff:.2e}")
    # At scale=1: comb = uncond + 1*(cond - uncond) = cond,
    # and the rescale factor cond/cond = 1.
    assert diff < 1e-5

    print("\n[done] All shape + pipeline checks passed.")


if __name__ == "__main__":
    demo()
