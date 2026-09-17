"""
Qwen Image Edit 2509 + InstantX ControlNet Union models.

Architecture:
  Backbone:     60 double-stream blocks, 3072 hidden, 3584 text dim (Qwen VL 7B)
  ControlNet:   5 double-stream blocks, trainable, same key structure
  VAE:          Qwen 3D VAE (AutoencoderKLWan), 16ch latent, 8x spatial
  Text encoder: Qwen 2.5-VL 7B LLM (pre-encoded; see encode_text_qwen.py)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from safetensors import safe_open


# ──────────────────────────────────────────────────────────────────────────────
# Constants
HIDDEN_DIM  = 3072
TEXT_DIM    = 3584
NUM_HEADS   = 24
HEAD_DIM    = 128   # HIDDEN_DIM / NUM_HEADS
MLP_DIM     = HIDDEN_DIM * 4   # 12288
PATCH_SIZE  = 2
LATENT_CH   = 16
PATCH_DIM   = LATENT_CH * PATCH_SIZE * PATCH_SIZE   # 64
FREQ_DIM    = 256


# ──────────────────────────────────────────────────────────────────────────────
# Utility: RMSNorm (functional)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rrms) * weight


def per_head_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x: [B, seq, H, head_dim], weight: [head_dim]"""
    rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rrms * weight


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal timestep embedding

def sinusoidal_embedding(t: torch.Tensor, dim: int = FREQ_DIM) -> torch.Tensor:
    """t: [B] float in [0, 1000].  Returns [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) *
                      torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args  = t[:, None].float() * freqs[None]
    emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb.to(t.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# 3-D RoPE with complex interleaved-pair format (matching official QwenEmbedRope)
# axes_dim=(16,56,56): splits head_dim=128 into temporal(16), height(56), width(56)
# complex format: view_as_complex on interleaved pairs, matches use_real=False

ROPE_AXES_DIM = (16, 56, 56)


def compute_rope_freqs_3d(
    h_patches: int, w_patches: int, t_patches: int = 1,
    head_dim: int = HEAD_DIM,
    axes_dim: tuple = ROPE_AXES_DIM,
    theta: float = 10000.0,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Returns complex RoPE freqs [T*H*W, head_dim//2] complex64.
    Interleaved-pair format: matches official apply_rotary_emb_qwen(use_real=False).
    """
    d_t = axes_dim[0] // 2   # 8  complex pairs for temporal
    d_h = axes_dim[1] // 2   # 28 complex pairs for height
    d_w = axes_dim[2] // 2   # 28 complex pairs for width

    def _axis_freqs(pos_len: int, n_complex: int) -> torch.Tensor:
        pos   = torch.arange(pos_len, device=device, dtype=torch.float32)
        omega = 1.0 / (theta ** (torch.arange(n_complex, device=device, dtype=torch.float32) / n_complex))
        angles = pos.unsqueeze(1) * omega.unsqueeze(0)   # [pos_len, n_complex]
        return torch.polar(torch.ones_like(angles), angles)   # complex64

    t_f = _axis_freqs(t_patches, d_t)   # [T, 8]
    h_f = _axis_freqs(h_patches, d_h)   # [H, 28]
    w_f = _axis_freqs(w_patches, d_w)   # [W, 28]

    T, H, W = t_patches, h_patches, w_patches
    t_3d = t_f[:, None, None, :].expand(T, H, W, d_t).reshape(T * H * W, d_t)
    h_3d = h_f[None, :, None, :].expand(T, H, W, d_h).reshape(T * H * W, d_h)
    w_3d = w_f[None, None, :, :].expand(T, H, W, d_w).reshape(T * H * W, d_w)

    return torch.cat([t_3d, h_3d, w_3d], dim=-1)   # [T*H*W, 64] complex64


def compute_text_rope_freqs(
    seq_len: int,
    head_dim: int = HEAD_DIM,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Text tokens at position (0,0,0) → exp(0j) = 1 → no rotation."""
    return torch.ones(seq_len, head_dim // 2, dtype=torch.complex64, device=device)


def apply_rope_complex(
    q: torch.Tensor, k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple:
    """
    q, k:  [B, seq, H, head_dim]
    freqs: [seq, head_dim//2] complex64
    Applies RoPE using complex multiplication on interleaved pairs.
    """
    def _rot(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        fc = f.unsqueeze(0).unsqueeze(2)   # [1, seq, 1, head_dim//2]
        return torch.view_as_real(xc * fc).flatten(3).to(x.dtype)
    return _rot(q, freqs), _rot(k, freqs)


# ──────────────────────────────────────────────────────────────────────────────
# Patch embedding helpers

def patchify(latent: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    latent: [B, C, T, H, W]  (3D VAE output, T=1 for images)
    Returns: [B, T*(H//p)*(W//p), C*T*p*p]
    """
    B, C, T, H, W = latent.shape
    p = patch_size
    x = latent.reshape(B, C, T, H // p, p, W // p, p)
    x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
    x = x.reshape(B, T * (H // p) * (W // p), C * p * p)
    return x


def unpatchify(tokens: torch.Tensor, h_patches: int, w_patches: int,
               t_patches: int = 1, patch_size: int = PATCH_SIZE,
               latent_ch: int = LATENT_CH) -> torch.Tensor:
    """tokens: [B, T*(H//p)*(W//p), C*p*p] → [B, C, T, H, W]"""
    B = tokens.shape[0]
    p = patch_size
    x = tokens.reshape(B, t_patches, h_patches, w_patches, latent_ch, p, p)
    x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
    x = x.reshape(B, latent_ch, t_patches, h_patches * p, w_patches * p)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# Functional double-stream block (used by both backbone and ControlNet)

def _lora_delta(x: torch.Tensor, ab, scale: float) -> torch.Tensor:
    """Low-rank residual: x @ A^T @ B^T * scale.  ab = (A[r,in], B[out,r])."""
    A, B = ab
    return ((x @ A.T.to(x.dtype)) @ B.T.to(x.dtype)) * scale


def double_stream_block_forward(
    img: torch.Tensor,    # [B, S_img, D]
    txt: torch.Tensor,    # [B, S_txt, D]
    vec: torch.Tensor,    # [B, D]
    w:   dict,            # weights with local keys
    img_freqs: torch.Tensor | None = None,   # [S_img, D//2] complex64
    txt_freqs: torch.Tensor | None = None,   # [S_txt, D//2] complex64
    lora: dict | None = None,                # {'to_q':(A,B),'to_k':..,'to_v':..,'to_out':..}
    lora_scale: float = 1.0,
) -> tuple:
    B, S_img, D = img.shape
    S_txt = txt.shape[1]

    # AdaLN modulation: SiLU already applied to vec inside img_mod Sequential
    # img_mod is Sequential(SiLU, Linear) so we apply SiLU here then Linear
    img_mods = F.silu(vec) @ w['img_mod.1.weight'].T + w['img_mod.1.bias']
    txt_mods = F.silu(vec) @ w['txt_mod.1.weight'].T + w['txt_mod.1.bias']

    img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = img_mods.chunk(6, dim=-1)
    txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = txt_mods.chunk(6, dim=-1)

    # Pre-norm + modulate
    img_n = F.layer_norm(img, [D]) * (1.0 + img_scale1[:, None]) + img_shift1[:, None]
    txt_n = F.layer_norm(txt, [D]) * (1.0 + txt_scale1[:, None]) + txt_shift1[:, None]

    # QKV projections (+ optional LoRA on the image stream)
    q_img = F.linear(img_n, w['attn.to_q.weight'], w['attn.to_q.bias'])
    k_img = F.linear(img_n, w['attn.to_k.weight'], w['attn.to_k.bias'])
    v_img = F.linear(img_n, w['attn.to_v.weight'], w['attn.to_v.bias'])
    if lora is not None:
        q_img = q_img + _lora_delta(img_n, lora['to_q'], lora_scale)
        k_img = k_img + _lora_delta(img_n, lora['to_k'], lora_scale)
        v_img = v_img + _lora_delta(img_n, lora['to_v'], lora_scale)
    Q_img = q_img.view(B, S_img, NUM_HEADS, HEAD_DIM)
    K_img = k_img.view(B, S_img, NUM_HEADS, HEAD_DIM)
    V_img = v_img.view(B, S_img, NUM_HEADS, HEAD_DIM)
    Q_txt = F.linear(txt_n, w['attn.add_q_proj.weight'], w['attn.add_q_proj.bias']).view(B, S_txt, NUM_HEADS, HEAD_DIM)
    K_txt = F.linear(txt_n, w['attn.add_k_proj.weight'], w['attn.add_k_proj.bias']).view(B, S_txt, NUM_HEADS, HEAD_DIM)
    V_txt = F.linear(txt_n, w['attn.add_v_proj.weight'], w['attn.add_v_proj.bias']).view(B, S_txt, NUM_HEADS, HEAD_DIM)

    # Per-head RMSNorm
    Q_img = per_head_rms_norm(Q_img, w['attn.norm_q.weight'])
    K_img = per_head_rms_norm(K_img, w['attn.norm_k.weight'])
    Q_txt = per_head_rms_norm(Q_txt, w['attn.norm_added_q.weight'])
    K_txt = per_head_rms_norm(K_txt, w['attn.norm_added_k.weight'])

    # 3D complex RoPE on image and text tokens
    if img_freqs is not None:
        Q_img, K_img = apply_rope_complex(Q_img, K_img, img_freqs.to(device=Q_img.device))
    if txt_freqs is not None:
        Q_txt, K_txt = apply_rope_complex(Q_txt, K_txt, txt_freqs.to(device=Q_txt.device))

    # Bidirectional attention (img+txt attend jointly)
    K = torch.cat([K_img, K_txt], dim=1)  # [B, S_img+S_txt, H, hd]
    V = torch.cat([V_img, V_txt], dim=1)
    Q = torch.cat([Q_img, Q_txt], dim=1)

    attn_out = F.scaled_dot_product_attention(
        Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
    ).transpose(1, 2)  # [B, S_img+S_txt, H, hd]

    attn_img = attn_out[:, :S_img].reshape(B, S_img, D)
    attn_txt = attn_out[:, S_img:].reshape(B, S_txt, D)

    img_attn = F.linear(attn_img, w['attn.to_out.0.weight'], w['attn.to_out.0.bias'])
    if lora is not None:
        img_attn = img_attn + _lora_delta(attn_img, lora['to_out'], lora_scale)
    txt_attn = F.linear(attn_txt, w['attn.to_add_out.weight'], w['attn.to_add_out.bias'])

    img = img + img_gate1[:, None] * img_attn
    txt = txt + txt_gate1[:, None] * txt_attn

    # MLP
    img_n2 = F.layer_norm(img, [D]) * (1.0 + img_scale2[:, None]) + img_shift2[:, None]
    txt_n2 = F.layer_norm(txt, [D]) * (1.0 + txt_scale2[:, None]) + txt_shift2[:, None]

    img_mlp = F.linear(
        F.gelu(F.linear(img_n2, w['img_mlp.net.0.proj.weight'], w['img_mlp.net.0.proj.bias'])),
        w['img_mlp.net.2.weight'], w['img_mlp.net.2.bias'])
    txt_mlp = F.linear(
        F.gelu(F.linear(txt_n2, w['txt_mlp.net.0.proj.weight'], w['txt_mlp.net.0.proj.bias'])),
        w['txt_mlp.net.2.weight'], w['txt_mlp.net.2.bias'])

    img = img + img_gate2[:, None] * img_mlp
    txt = txt + txt_gate2[:, None] * txt_mlp
    return img, txt


# ──────────────────────────────────────────────────────────────────────────────
# Backbone (frozen, functional)

class QwenBackbone:
    """
    Frozen backbone. Weights stored as fp8 tensors on GPU.
    Each block cast to bf16 just-in-time during forward.
    """
    NUM_BLOCKS = 60

    def __init__(self, path: str, device: torch.device):
        self.device = device
        self._w: dict[str, torch.Tensor] = {}
        print(f'[Backbone] Loading fp8 weights from {path} …')
        with safe_open(path, framework='pt', device='cpu') as f:
            for k in f.keys():
                if k.startswith('model.diffusion_model.'):
                    self._w[k] = f.get_tensor(k).to(device)
        print(f'[Backbone] Loaded {len(self._w)} tensors on {device}')

    def _get(self, key: str) -> torch.Tensor:
        return self._w[f'model.diffusion_model.{key}'].to(torch.bfloat16)

    def _block_w(self, i: int) -> dict:
        prefix = f'model.diffusion_model.transformer_blocks.{i}.'
        plen = len(prefix)
        return {k[plen:]: v.to(torch.bfloat16)
                for k, v in self._w.items() if k.startswith(prefix)}

    def forward(
        self,
        img_packed:  torch.Tensor,    # [B, S_noisy, PATCH_DIM]
        txt_emb:     torch.Tensor,    # [B, S_txt, TEXT_DIM]
        t_sigma:     torch.Tensor,    # [B]  in [0,1]
        controlnet_residuals: list,   # list of [B, S_noisy, HIDDEN_DIM]
        img_freqs: torch.Tensor | None = None,   # [S_noisy+S_ref, HEAD_DIM//2] complex64
        txt_freqs: torch.Tensor | None = None,   # [S_txt(+gaze), HEAD_DIM//2] complex64
        ref_packed: torch.Tensor | None = None,  # [B, S_ref, PATCH_DIM] clean reference
        gaze_tokens: torch.Tensor | None = None, # [B, N_gaze, HIDDEN_DIM] trainable gaze tokens
        gaze_vec: torch.Tensor | None = None,    # [B, HIDDEN_DIM] adaLN modulation add-on (gaze)
        lora_blocks: list | None = None,         # per-block LoRA dicts (len NUM_BLOCKS)
        lora_scale: float = 1.0,
    ) -> torch.Tensor:                # [B, S_noisy, PATCH_DIM]
        """
        Backbone forward with optional reference image input (Qwen Image Edit).
        When ref_packed is given, ref tokens are concatenated to noisy tokens
        along the sequence dim (ref gets a different temporal RoPE position),
        letting self-attention preserve identity from the reference.

        ControlNet residuals are added ONLY to the noisy half (ref tokens are
        unaltered by ControlNet).  The output returns only the noisy half.
        """
        D = HIDDEN_DIM
        S_noisy = img_packed.shape[1]
        has_ref = ref_packed is not None

        # Image patch embedding (under no_grad since no residuals injected here)
        with torch.no_grad():
            img = F.linear(img_packed, self._get('img_in.weight'), self._get('img_in.bias'))
            if has_ref:
                ref_emb = F.linear(ref_packed, self._get('img_in.weight'), self._get('img_in.bias'))
                img = torch.cat([img, ref_emb], dim=1)   # [B, S_noisy+S_ref, D]
            txt_n = rms_norm(txt_emb, self._get('txt_norm.weight'))
            txt   = F.linear(txt_n, self._get('txt_in.weight'), self._get('txt_in.bias'))
            t_emb = sinusoidal_embedding(t_sigma * 1000.0, FREQ_DIM).to(img.device, img.dtype)
            vec   = F.silu(F.linear(t_emb,
                                     self._get('time_text_embed.timestep_embedder.linear_1.weight'),
                                     self._get('time_text_embed.timestep_embedder.linear_1.bias')))
            vec   = F.linear(vec,
                              self._get('time_text_embed.timestep_embedder.linear_2.weight'),
                              self._get('time_text_embed.timestep_embedder.linear_2.bias'))

        # All blocks use gradient checkpointing so that:
        # - bf16 weight dicts are created inside the closure and freed after each block
        # - Only one block's bf16 weights reside in GPU memory at a time
        # - Activations are recomputed during backward rather than stored
        # - Backbone weight tensors (requires_grad=False) accumulate no gradients
        # - After each residual addition img.requires_grad=True → gradient flows to ControlNet
        # Append trainable gaze tokens to the text stream (outside no_grad so their
        # gradient is preserved).  They are already in HIDDEN_DIM space.
        if gaze_tokens is not None:
            txt = torch.cat([txt, gaze_tokens.to(txt.dtype)], dim=1)

        # adaLN-Zero gaze injection: add gaze modulation to `vec` (computed under
        # no_grad above) OUTSIDE the no_grad block so gradient flows to the gaze
        # encoder.  `vec` drives every block's scale/shift/gate + final norm_out,
        # i.e. gaze conditions all 60 layers globally (DiT class-conditioning style).
        if gaze_vec is not None:
            vec = vec + gaze_vec.to(vec.dtype)

        n_cn = len(controlnet_residuals)
        # Official injection pattern: each residual spans ceil(60/5)=12 consecutive blocks
        interval = math.ceil(self.NUM_BLOCKS / n_cn) if n_cn > 0 else 1

        for i in range(self.NUM_BLOCKS):
            def _make_block_fn(block_idx, _vec, _img_freqs, _txt_freqs, _lora):
                def _block_fn(img, txt):
                    bw = self._block_w(block_idx)
                    return double_stream_block_forward(img, txt, _vec, bw, _img_freqs, _txt_freqs,
                                                       lora=_lora, lora_scale=lora_scale)
                return _block_fn

            _lora = lora_blocks[i] if lora_blocks is not None else None
            img, txt = grad_ckpt.checkpoint(
                _make_block_fn(i, vec, img_freqs, txt_freqs, _lora),
                img, txt,
                use_reentrant=False,
            )
            if n_cn > 0:
                cn_idx = min(i // interval, n_cn - 1)
                if has_ref:
                    # Pad residual with zeros for the ref-token slice
                    B, S_total, H = img.shape
                    pad = torch.zeros(B, S_total - S_noisy, H, device=img.device, dtype=img.dtype)
                    res = torch.cat([controlnet_residuals[cn_idx], pad], dim=1)
                    img = img + res
                else:
                    img = img + controlnet_residuals[cn_idx]

        # Take only the noisy half — ref tokens are auxiliary
        if has_ref:
            img = img[:, :S_noisy, :]

        # Final AdaLN + project (outside no_grad to preserve gradient path from residuals)
        norm_emb = F.linear(F.silu(vec),
                             self._get('norm_out.linear.weight'),
                             self._get('norm_out.linear.bias'))
        scale, shift = norm_emb.chunk(2, dim=-1)
        img = F.layer_norm(img, [D]) * (1.0 + scale[:, None]) + shift[:, None]
        return F.linear(img, self._get('proj_out.weight'), self._get('proj_out.bias'))


# ──────────────────────────────────────────────────────────────────────────────
# ControlNet block sub-modules (matching safetensors key names)

class _GELUProj(nn.Module):
    """Wrapper with .proj attribute for key 'img_mlp.net.0.proj.*'"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x))


class _DoubleStreamMLP(nn.Module):
    """MLP matching key structure: net.0.proj.*, net.2.*"""
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.net = nn.ModuleList([
            _GELUProj(dim, mlp_dim),
            nn.Identity(),
            nn.Linear(mlp_dim, dim),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[0](x))


class _DoubleStreamAttn(nn.Module):
    """Attention sub-module matching ControlNet key names."""
    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.to_q   = nn.Linear(dim, dim)
        self.to_k   = nn.Linear(dim, dim)
        self.to_v   = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.to_add_out = nn.Linear(dim, dim)
        # Per-head RMSNorm weights (not nn.Linear; stored as Parameters)
        self.norm_q       = nn.Parameter(torch.ones(head_dim))
        self.norm_k       = nn.Parameter(torch.ones(head_dim))
        self.norm_added_q = nn.Parameter(torch.ones(head_dim))
        self.norm_added_k = nn.Parameter(torch.ones(head_dim))


class QwenControlNetBlock(nn.Module):
    """Single double-stream block for ControlNet."""
    def __init__(self, dim: int = HIDDEN_DIM, n_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM):
        super().__init__()
        # img_mod / txt_mod: Sequential(SiLU [index 0], Linear [index 1])
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.attn    = _DoubleStreamAttn(dim, n_heads, head_dim)
        self.img_mlp = _DoubleStreamMLP(dim, dim * 4)
        self.txt_mlp = _DoubleStreamMLP(dim, dim * 4)
        self._n_heads  = n_heads
        self._head_dim = head_dim

    def _w(self) -> dict:
        return {
            'img_mod.1.weight': self.img_mod[1].weight,
            'img_mod.1.bias':   self.img_mod[1].bias,
            'txt_mod.1.weight': self.txt_mod[1].weight,
            'txt_mod.1.bias':   self.txt_mod[1].bias,
            'attn.to_q.weight':       self.attn.to_q.weight,
            'attn.to_q.bias':         self.attn.to_q.bias,
            'attn.to_k.weight':       self.attn.to_k.weight,
            'attn.to_k.bias':         self.attn.to_k.bias,
            'attn.to_v.weight':       self.attn.to_v.weight,
            'attn.to_v.bias':         self.attn.to_v.bias,
            'attn.to_out.0.weight':   self.attn.to_out[0].weight,
            'attn.to_out.0.bias':     self.attn.to_out[0].bias,
            'attn.add_q_proj.weight': self.attn.add_q_proj.weight,
            'attn.add_q_proj.bias':   self.attn.add_q_proj.bias,
            'attn.add_k_proj.weight': self.attn.add_k_proj.weight,
            'attn.add_k_proj.bias':   self.attn.add_k_proj.bias,
            'attn.add_v_proj.weight': self.attn.add_v_proj.weight,
            'attn.add_v_proj.bias':   self.attn.add_v_proj.bias,
            'attn.to_add_out.weight': self.attn.to_add_out.weight,
            'attn.to_add_out.bias':   self.attn.to_add_out.bias,
            'attn.norm_q.weight':        self.attn.norm_q,
            'attn.norm_k.weight':        self.attn.norm_k,
            'attn.norm_added_q.weight':  self.attn.norm_added_q,
            'attn.norm_added_k.weight':  self.attn.norm_added_k,
            'img_mlp.net.0.proj.weight': self.img_mlp.net[0].proj.weight,
            'img_mlp.net.0.proj.bias':   self.img_mlp.net[0].proj.bias,
            'img_mlp.net.2.weight':      self.img_mlp.net[2].weight,
            'img_mlp.net.2.bias':        self.img_mlp.net[2].bias,
            'txt_mlp.net.0.proj.weight': self.txt_mlp.net[0].proj.weight,
            'txt_mlp.net.0.proj.bias':   self.txt_mlp.net[0].proj.bias,
            'txt_mlp.net.2.weight':      self.txt_mlp.net[2].weight,
            'txt_mlp.net.2.bias':        self.txt_mlp.net[2].bias,
        }

    def forward(self, img, txt, vec, img_freqs=None, txt_freqs=None):
        return double_stream_block_forward(img, txt, vec, self._w(), img_freqs, txt_freqs)


# ──────────────────────────────────────────────────────────────────────────────
# ControlNet (trainable nn.Module)

class QwenControlNet(nn.Module):
    """InstantX ControlNet for Qwen Image Edit 2509.  5 blocks."""
    NUM_BLOCKS = 5

    def __init__(self, dim: int = HIDDEN_DIM, text_dim: int = TEXT_DIM):
        super().__init__()
        self.img_in               = nn.Linear(PATCH_DIM, dim)
        self.controlnet_x_embedder = nn.Linear(PATCH_DIM, dim)
        self.txt_norm             = nn.Parameter(torch.ones(text_dim))
        self.txt_in               = nn.Linear(text_dim, dim)

        # Timestep embedder: Sequential(Linear [index 0 = linear_1], …)
        # We store as a nested Module to match key names
        self.time_text_embed = _TimestepEmbedModule(dim=dim, freq_dim=FREQ_DIM)

        self.transformer_blocks = nn.ModuleList(
            [QwenControlNetBlock() for _ in range(self.NUM_BLOCKS)])
        self.controlnet_blocks = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(self.NUM_BLOCKS)])

    def forward(
        self,
        img_packed:  torch.Tensor,   # [B, S_img, PATCH_DIM]
        cond_packed: torch.Tensor,   # [B, S_img, PATCH_DIM]
        txt_emb:     torch.Tensor,   # [B, S_txt, TEXT_DIM]
        t_sigma:     torch.Tensor,   # [B]
        img_freqs: torch.Tensor | None = None,   # [S_img, HEAD_DIM//2] complex64
        txt_freqs: torch.Tensor | None = None,   # [S_txt, HEAD_DIM//2] complex64
    ) -> list:                       # [NUM_BLOCKS] × [B, S_img, HIDDEN_DIM]

        img = self.img_in(img_packed) + self.controlnet_x_embedder(cond_packed)
        txt = self.txt_in(rms_norm(txt_emb, self.txt_norm))
        vec = self.time_text_embed(t_sigma.to(img.device, img.dtype))

        residuals = []
        for i, block in enumerate(self.transformer_blocks):
            img, txt = block(img, txt, vec, img_freqs, txt_freqs)
            residuals.append(self.controlnet_blocks[i](img))
        return residuals

    def load_pretrained(self, path: str):
        """Load weights from Qwen-Image-InstantX-ControlNet-Union.safetensors."""
        sd: dict[str, torch.Tensor] = {}
        with safe_open(path, framework='pt', device='cpu') as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k).to(torch.bfloat16)

        own_sd = {k: v for k, v in self.named_parameters()}
        own_sd.update({k: v for k, v in self.named_buffers()})

        loaded = set()
        for file_key, tensor in sd.items():
            if file_key not in own_sd:
                # Try to find a match by navigating the module tree
                try:
                    self._set_nested(file_key, tensor)
                    loaded.add(file_key)
                except Exception:
                    pass
            else:
                own_sd[file_key].data.copy_(tensor)
                loaded.add(file_key)

        missing = [k for k in sd if k not in loaded]
        print(f'[ControlNet] Loaded {len(loaded)}/{len(sd)} keys from {path}')
        if missing:
            print(f'  Not matched: {missing[:5]}')

    def _set_nested(self, key: str, tensor: torch.Tensor):
        """Navigate dot-separated key and set the parameter value."""
        parts = key.split('.')
        obj   = self
        for i, p in enumerate(parts[:-1]):
            child = obj[int(p)] if p.isdigit() else getattr(obj, p)
            # If the current child is already a Parameter (e.g. norm_q is a bare Parameter)
            # and the remaining path is just '.weight', apply directly.
            if isinstance(child, nn.Parameter) and i == len(parts) - 2 and parts[-1] == 'weight':
                child.data.copy_(tensor)
                return
            obj = child
        attr = parts[-1]
        param = obj[int(attr)] if attr.isdigit() else getattr(obj, attr)
        if isinstance(param, nn.Parameter):
            param.data.copy_(tensor)
        else:
            raise AttributeError(f'Not a Parameter: {key}')


class _TimestepEmbedModule(nn.Module):
    """Matches key structure time_text_embed.timestep_embedder.linear_1/2"""
    def __init__(self, dim: int = HIDDEN_DIM, freq_dim: int = FREQ_DIM):
        super().__init__()
        self._freq_dim = freq_dim
        self.timestep_embedder = _TimestepMLP(dim, freq_dim)

    def forward(self, t_sigma: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(t_sigma)


class _TimestepMLP(nn.Module):
    """Sinusoidal → linear_1 → SiLU → linear_2"""
    def __init__(self, dim: int, freq_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(freq_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self._freq_dim = freq_dim

    def forward(self, t_sigma: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t_sigma * 1000.0, self._freq_dim).to(dtype=self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(t_emb)))


# ──────────────────────────────────────────────────────────────────────────────
# VAE loader

def _vae_file_to_model_key(fk: str) -> str | None:
    """Map a Qwen VAE safetensors file key → AutoencoderKLWan model key."""
    # Residual subkey mapping
    _res_sub = {
        'residual.0.gamma': 'norm1.gamma',
        'residual.2.weight': 'conv1.weight',
        'residual.2.bias': 'conv1.bias',
        'residual.3.gamma': 'norm2.gamma',
        'residual.6.weight': 'conv2.weight',
        'residual.6.bias': 'conv2.bias',
    }

    def _resnet_sub(sub: str) -> str | None:
        if sub in _res_sub:
            return _res_sub[sub]
        if sub.startswith('shortcut.'):
            return 'conv_shortcut.' + sub[len('shortcut.'):]
        return sub  # pass-through (resample.*, time_conv.*)

    # quant / post_quant
    if fk.startswith('conv1.'):
        return 'quant_conv.' + fk[len('conv1.'):]
    if fk.startswith('conv2.'):
        return 'post_quant_conv.' + fk[len('conv2.'):]

    # Encoder: conv_in, norm_out, conv_out
    if fk.startswith('encoder.conv1.'):
        return 'encoder.conv_in.' + fk[len('encoder.conv1.'):]
    if fk == 'encoder.head.0.gamma':
        return 'encoder.norm_out.gamma'
    if fk.startswith('encoder.head.2.'):
        return 'encoder.conv_out.' + fk[len('encoder.head.2.'):]

    # Encoder mid_block: middle.0 → resnets.0, middle.1 → attentions.0, middle.2 → resnets.1
    if fk.startswith('encoder.middle.'):
        rest = fk[len('encoder.middle.'):]
        idx_s, _, sub = rest.partition('.')
        idx = int(idx_s)
        if idx in (0, 2):
            rn_idx = 0 if idx == 0 else 1
            return f'encoder.mid_block.resnets.{rn_idx}.{_resnet_sub(sub)}'
        else:  # idx == 1 (attention)
            return f'encoder.mid_block.attentions.0.{sub}'

    # Encoder down_blocks: downsamples.N → down_blocks.N (1:1)
    if fk.startswith('encoder.downsamples.'):
        rest = fk[len('encoder.downsamples.'):]
        n_s, _, sub = rest.partition('.')
        return f'encoder.down_blocks.{n_s}.{_resnet_sub(sub)}'

    # Decoder: conv_in, norm_out, conv_out
    if fk.startswith('decoder.conv1.'):
        return 'decoder.conv_in.' + fk[len('decoder.conv1.'):]
    if fk == 'decoder.head.0.gamma':
        return 'decoder.norm_out.gamma'
    if fk.startswith('decoder.head.2.'):
        return 'decoder.conv_out.' + fk[len('decoder.head.2.'):]

    # Decoder mid_block
    if fk.startswith('decoder.middle.'):
        rest = fk[len('decoder.middle.'):]
        idx_s, _, sub = rest.partition('.')
        idx = int(idx_s)
        if idx in (0, 2):
            rn_idx = 0 if idx == 0 else 1
            return f'decoder.mid_block.resnets.{rn_idx}.{_resnet_sub(sub)}'
        else:
            return f'decoder.mid_block.attentions.0.{sub}'

    # Decoder up_blocks: upsamples.K → up_blocks.(K//4).resnets.(K%4) or upsamplers.0
    if fk.startswith('decoder.upsamples.'):
        rest = fk[len('decoder.upsamples.'):]
        k_s, _, sub = rest.partition('.')
        k = int(k_s)
        block = k // 4
        local = k % 4
        if local == 3:   # upsampler (resample / time_conv)
            return f'decoder.up_blocks.{block}.upsamplers.0.{sub}'
        else:            # resnet
            return f'decoder.up_blocks.{block}.resnets.{local}.{_resnet_sub(sub)}'

    return None   # unmapped


def load_qwen_vae(path: str, device: torch.device = torch.device('cpu')):
    """Load Qwen 3D VAE.  Returns frozen model in eval mode."""
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan(
        base_dim=96, z_dim=LATENT_CH, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True],   # typo preserved from diffusers
        dropout=0.0,
        latents_mean=[0.0] * LATENT_CH,
        latents_std=[1.0]  * LATENT_CH,
    )

    sd_file: dict[str, torch.Tensor] = {}
    with safe_open(path, framework='pt', device='cpu') as f:
        for k in f.keys():
            sd_file[k] = f.get_tensor(k)

    # Build remapped state dict
    remapped: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for fk, v in sd_file.items():
        mk = _vae_file_to_model_key(fk)
        if mk is not None:
            remapped[mk] = v
        else:
            unmapped.append(fk)

    if unmapped:
        print(f'[VAE] {len(unmapped)} unmapped file keys (first few): {unmapped[:3]}')

    res = vae.load_state_dict(remapped, strict=False)
    n_ok = len(sd_file) - len(unmapped)
    print(f'[VAE] Loaded {n_ok}/{len(sd_file)} keys  '
          f'missing={len(res.missing_keys)}  unexpected={len(res.unexpected_keys)}')

    vae = vae.to(device=device, dtype=torch.bfloat16).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ──────────────────────────────────────────────────────────────────────────────
# VAE encode / decode

@torch.no_grad()
def vae_encode(vae, image_rgb: torch.Tensor) -> torch.Tensor:
    """image_rgb: [B, 3, H, W] in [-1,1] → [B, 16, 1, H//8, W//8]"""
    x = image_rgb.unsqueeze(2)   # [B, 3, 1, H, W]
    return vae.encode(x).latent_dist.sample()


@torch.no_grad()
def vae_decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """latent: [B, 16, 1, H//8, W//8] → [B, 3, H, W] in [-1,1]"""
    out = vae.decode(latent).sample   # [B, 3, 1, H, W]
    return out[:, :, 0]
