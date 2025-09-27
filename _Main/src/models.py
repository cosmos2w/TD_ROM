


from __future__ import annotations
from torchdiffeq import odeint
from typing import Tuple, Callable, Optional
from torch.utils.checkpoint import checkpoint
from collections import deque

import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.distributions as dist
import os, math
import numpy as np

from einops import repeat
from fairscale.nn import checkpoint_wrapper

#------------------------------------------
# These parts are for TD-ROM
#------------------------------------------

def apply_rope(x: torch.Tensor,
               pos: torch.Tensor,
               base: float = 10000.0) -> torch.Tensor:
    """
    Rotary Position Embedding.
    x   : (B, L, D)            - token features, D must be even
    pos : (B, L, 1) or (B, L)  - scalar position of every token
    """
    B, L, D = x.shape
    assert D % 2 == 0, "RoPE needs an even hidden size"
    half_D = D // 2

    # frequencies 1 / base^{2k/d}
    inv_freq = 1.0 / (base ** (torch.arange(0, half_D, device=x.device).float() / half_D))  # (D/2,)

    # broadcast positions → (B, L, D/2)
    theta = pos.float() * inv_freq            # (B, L, D/2)
    sin, cos = torch.sin(theta), torch.cos(theta)

    # split the embedding in two blocks [0:half_D | half_D:]
    x1, x2 = x[..., :half_D], x[..., half_D:]  # each (B, L, D/2)

    # apply planar rotation
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x2 * cos + x1 * sin
    return torch.cat([rot_x1, rot_x2], dim=-1)

class MLP(nn.Module):
    def __init__(self, dims, activation='relu', use_bias=True, final_activation=None):
        """
        dims (list of int): List such that dims = [n_input, hidden1, ..., n_output].
        activation (str):   Activation for the hidden layers ("relu", "tanh", etc.).
        use_bias (bool):    Whether to use bias in each linear layer.
        final_activation (str or None): Final activation function (if any) for the last layer.
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        act_fn = self.get_activation(activation)
        final_act_fn = self.get_activation(final_activation) if final_activation is not None else None
        
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1], bias=use_bias))
            if i < len(dims)-2:
                self.activations.append(act_fn)
            else:
                self.activations.append(final_act_fn)
                
    def get_activation(self, act_str):
        if act_str is None:
            return None
        if act_str.lower() == 'relu':
            return nn.ReLU()
        elif act_str.lower() == 'tanh':
            return nn.Tanh()
        elif act_str.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {act_str}")

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            if act is not None:
                x = act(x)
        return x

class GEGLU(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.proj = nn.Linear(d_model, mult * d_model * 2)  # 2× for gate
        self.out  = nn.Linear(mult * d_model, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)   # each (…,4d)
        return self.out(a * self.gelu(b))

class FourierEmbedding(nn.Module):
    """
    phi(x) = [sin(2pi B x) , cos(2pi B x)], B ∈ R^{d * M}
    """
    def __init__(self, in_dim: int = 2, num_frequencies: int = 64,
                 learnable: bool = True, sigma: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.B = nn.Parameter(B, requires_grad=learnable)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * xy @ self.B          # (..., M)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class _CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)
    def forward(self, q, k, v):
        y, _ = self.attn(q, k, v, need_weights=False)
        return y

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        layers: int = 1,               # number of repeated applications
        use_layernorm: bool = False,   # stability when layers > 1
        residual: bool = False,        # q + attn_out
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layers = layers
        self.residual = residual
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()

    def forward(self, q, k, v):
        """
        q: (B, Tq, C), k: (B, Tk, C), v: (B, Tk, C)
        returns: (B, Tq, C)
        """
        x = q
        for _ in range(self.layers):
            out, _ = self.attn(x, k, v, need_weights=False)
            if self.residual:
                out = x + self.dropout(out)
            out = self.norm(out)
            x = out
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attn(x, x, pad_mask=pad_mask, attn_mask=attn_mask)

def latent_block(dim, num_heads, dropout):
    return nn.TransformerEncoderLayer(dim,
                                      num_heads,
                                      dropout=dropout,
                                      batch_first=True,
                                      activation="gelu")

# ----------------------------------------------------------------------------- 
# Cross–attention over sensors (+CLS)
# -----------------------------------------------------------------------------
class FourierTransformerSpatialEncoder(nn.Module):
    def __init__(self,
                 All_dim         : int,   # = 128,
                 num_heads       : int,   # = 4,
                 latent_layers   : int,   # = 2,
                 N_channels      : int,

                 num_freqs       : int   = 64,
                 use_temporal    : bool  = False,
                 latent_tokens   : int   = 8,
                 pooling         : str   = "cls",  # "mean" or "cls"
                 Token_dim       : int   = 128,
                 rope_base       : float = 10000.0,
                 dropout         : float = 0.0,
                 ):
        """
        pooling:
            "mean" - average latent tokens (old behaviour)
            "cls"  - prepend a learnable CLS token and return it

        """
        super().__init__()
        assert pooling in ("mean", "cls", "none")
        self.pooling      = pooling
        self.use_temporal = use_temporal
        self.All_dim      = All_dim
        self.N_channels   = N_channels
        self.num_heads    = num_heads
        self.Token_dim    = Token_dim
        self.rope_base    = rope_base

        # Fourier features → Linear
        self.pos_embed  = FourierEmbedding(2, num_freqs, learnable=True)
        self.pos_linear = nn.Linear(2 * num_freqs, All_dim)
        self.pos_norm   = nn.LayerNorm(All_dim)

        # Value projection
        self.val_linear = nn.Linear(N_channels, All_dim)
        self.val_norm   = nn.LayerNorm(All_dim)

        # Learnable latent tokens
        self.L = latent_tokens
        self.latent_param = nn.Parameter(
            torch.randn(1, self.L, All_dim) * 0.02
        )

        if pooling == "cls":
            # one extra token, fixed to zeros at start
            self.cls_token = nn.Parameter(torch.zeros(1, 1, All_dim))
            self.cls_index = 0
        else:
            self.cls_token = None
            self.cls_index = None

        # 4.  Attention modules
        self.cross_attn   = CrossAttention(All_dim, num_heads, dropout)
        self.cross_norm   = nn.LayerNorm(All_dim)

        self.latent_mixer = nn.TransformerEncoder(
            latent_block(All_dim, num_heads, dropout),
            num_layers=latent_layers
        )
        if use_temporal:
            self.temporal_mixer = nn.TransformerEncoder(
                latent_block(All_dim, num_heads, dropout),
                num_layers=latent_layers
            )

        self.time_linear  = nn.Linear(1, All_dim)
        self.token_projection = nn.Linear(Token_dim, All_dim)

    # -----------------------------------------------------------------
    # Rotary positional embedding along the temporal axis
    # -----------------------------------------------------------------
    def apply_rope_t(self, x, t):
        """
        x : (B, T, D),   t : (B, T, 1)  ∈ [0,1]
        """
        d  = x.size(-1) // 2
        freq = self.rope_base ** (-torch.arange(d, device=x.device) / d)  # (d,)
        angle = t * freq                                                  # (B,T,d)
        sin, cos = angle.sin(), angle.cos()
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], dim=-1)

    def forward(self, coords_tuv, U):
        """
        coords_tuv : (B, T, N, 2 + N_channels + 1) = (x, y, values..., t)
        """
        B, T, N, _ = coords_tuv.shape
        xy   = coords_tuv[..., 0:2]                            # (B,T,N,2)
        val  = coords_tuv[..., 2:(2+self.N_channels)]          # (B,T,N,N_channels)
        tt   = coords_tuv[..., -1:]                            # (B,T,N,1), constant in N

        # --- sensor token ---------------------------------------------------------

        pos_tok = self.pos_linear(self.pos_embed(xy))
        val_tok = self.val_linear(val)

        tok     = pos_tok + val_tok                                          # (B,T,N,D)
        tok  = tok.reshape(B * T, N, self.All_dim)                           # merge time

        # --- latent set -----------------------------------------------------------
        lat = self.latent_param.expand(B * T, -1, -1).clone()             # (B*T,L,D)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B * T, -1, -1)
            lat = torch.cat([cls, lat], dim=1)

        t0  = coords_tuv[..., -1:, 0]                                # (B,T,1)
        t0  = self.time_linear(t0).repeat_interleave(lat.size(1), 1) # (B,T*L,D)
        lat = lat + t0.view(B * T, lat.size(1), -1)

        # cross-attention
        lat = lat + self.cross_attn(lat, tok, tok)

        lat = self.latent_mixer(lat)

        # --- pooling --------------------------------------------------------------
        if self.pooling == "mean":
            lat_vec = lat.mean(dim=1)  # (B*T, D)
        elif self.pooling == "cls":
            lat_vec = lat[:, self.cls_index, :]  # (B*T, D)
        else:  # "none"
            lat_vec = lat  # (B*T, L, D)

        lat_vec = lat_vec.view(B, T, -1)

        if self.pooling == "none":
            lat_vec = lat_vec.view(B, T, lat.shape[1], self.All_dim)       # (B, T, L, D)
        else:
            lat_vec = lat_vec.view(B, T, self.All_dim)                     # (B, T, D)

        # --- temporal modelling ---------------------------------------------------
        if self.use_temporal and T > 1:
            t0 = tt[:, :, 0, :]                                            # (B,T,1)
            if self.pooling == "none":
                _, _, L, D = lat_vec.shape
                lat_vec = lat_vec.permute(0, 2, 1, 3).contiguous().view(B * L, T, D)  # (B*L, T, D)
                t0 = t0.unsqueeze(1).expand(-1, L, -1, -1).contiguous().view(B * L, T, 1)
            lat_vec = self.apply_rope_t(lat_vec, t0)
            lat_vec = self.temporal_mixer(lat_vec)
            if self.pooling == "none":
                lat_vec = lat_vec.view(B, L, T, D).permute(0, 2, 1, 3)  # (B, T, L, D)

        return lat_vec

# Revised 09.14
# --------------------------------------------------
#  Small re-usable building blocks
# --------------------------------------------------
def build_pos_value_proj(N_channels: int,
                         num_freqs: int,
                         All_dim: int) -> nn.ModuleDict:
    return nn.ModuleDict({
        "pos_embed": FourierEmbedding(2, num_freqs, learnable=True),
        "pos_linear": nn.Linear(2 * num_freqs, All_dim),
        "pos_norm": nn.RMSNorm(All_dim),
        "val_linear": nn.Linear(N_channels, All_dim),
        "val_norm": nn.RMSNorm(All_dim),
    })

def build_latent_bank(All_dim: int,
                      latent_tokens: int,
                      with_cls: bool = True) -> nn.ParameterDict:
    p = {}
    p["latent_param"] = nn.Parameter(torch.randn(1, latent_tokens, All_dim) * 0.02)
    if with_cls:
        p["cls_token"]  = nn.Parameter(torch.zeros(1, 1, All_dim))
    return nn.ParameterDict(p)

def build_transformer_stack(All_dim: int,
                            num_heads: int,
                            num_layers: int,
                            dropout: float = 0.) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(
        latent_block(All_dim, num_heads, dropout),
        num_layers=num_layers
    )

# --------------------------------------------------
#  Stand-alone Domain Adaptive Encoder Module
# --------------------------------------------------
class DomainAdaptiveEncoder(nn.Module):

    def __init__(self,
                 All_dim       : int,
                 num_heads     : int,
                 latent_layers : int,
                 N_channels    : int,
                 num_freqs     : int = 64,
                 latent_tokens : int = 8,
                 pooling       : str = "none",      # currently ignored
                 *,
                 retain_cls    : bool = False
                 ):
        super().__init__()
        assert pooling in ("mean", "cls", "none")

        # 1. Positional + value projection sub-modules ---------------------------
        self.embed = build_pos_value_proj(N_channels, num_freqs, All_dim)

        # 2. Latent / CLS parameters --------------------------------------------
        self.latents = build_latent_bank(All_dim, latent_tokens, with_cls=True)

        # 3. Attention blocks ----------------------------------------------------
        self.cross_attn            = CrossAttention(All_dim, num_heads, dropout = 0.0,
                                                    layers = latent_layers
                                                    )
        self.cross_norm            = nn.RMSNorm(All_dim)
        self.cross_latent_mixer    = build_transformer_stack(All_dim, num_heads,
                                                             latent_layers)

        self.token_to_latent       = CrossAttention(All_dim, num_heads, dropout = 0.0,
                                                    layers = latent_layers
                                                    )
        self.token_to_latent_norm  = nn.RMSNorm(All_dim)
        self.token_to_latent_mixer = build_transformer_stack(All_dim, num_heads,
                                                             latent_layers)

        # -----------------------------------------------------------------------
        self.retain_cls = retain_cls
        print(f'self.retain_cls is {self.retain_cls} ! ')

    def forward(self,
                coords_tuv   : torch.Tensor,                     # (B,T,N, 2+C)
                U            : torch.Tensor,                     # (unused here)
                original_phi : Optional[torch.Tensor] = None     # (B,S)
                ):

        B, T, N_inp, _ = coords_tuv.shape

        # ------------------------------------------------------------------ 
        # Split coordinates and sensor values
        xy   = coords_tuv[..., 0:2]                                            # (B,T,N,2)
        C_in = self.embed['val_linear'].in_features
        vals = coords_tuv[..., 2:2 + C_in]                                     # (B,T,N,C)

        # Positional & value embeddings
        pos_tok = self.embed['pos_linear'](self.embed['pos_embed'](xy))
        pos_tok = self.embed['pos_norm'](pos_tok)

        val_tok = self.embed['val_linear'](vals)
        val_tok = self.embed['val_norm'](val_tok)

        tok      = pos_tok + val_tok                                           # (B,T,N,D)
        tok_flat = tok.reshape(B * T, N_inp, -1)                               # (B*T,N,D)

        # ------------------------------------------------------------------ 
        # Prepare latent + CLS tokens
        if self.retain_cls:
            cls = self.latents['cls_token'].expand(B * T, 1, -1)                   # (B*T,1,D)
            lat = self.latents['latent_param'].expand(B * T, -1, -1)               # (B*T,L,D)
            lat = torch.cat([cls, lat], dim=1)                                     # (B*T,1+L,D)

        # Cross-attention: sensor tokens -> latent tokens
        lat = self.cross_norm(lat)
        lat = lat + self.cross_attn(lat, tok_flat, tok_flat)
        lat = lat + self.cross_latent_mixer(lat)

        # ------------------------------------------------------------------ 
        # Feed refined latent information back into sensor tokens
        tok_flat_up  = self.token_to_latent_norm(tok_flat)
        tok_flat_up  = tok_flat_up + self.token_to_latent(tok_flat_up, lat, lat)
        tok_flat_out = tok_flat_up + self.token_to_latent_mixer(tok_flat_up)

        # tok_flat_out = tok_flat_up + self.token_to_latent(tok_flat_up, lat, lat)

        # ------------------------------------------------------------------ 
        # Optionally keep CLS as part of the output token set
        if self.retain_cls:
            cls_upd   = lat[:, :1, :]                                          # (B*T,1,D)
            tok_flat_out = torch.cat([cls_upd, tok_flat_out], dim=1)           # (B*T,1+N,D)

        # ------------------------------------------------------------------ 
        # Reshape back to (B,T,⋯)
        S        = tok_flat_out.size(1)
        lat_vec  = tok_flat_out.view(B, T, S, -1)                              # (B,T,S,D)

        mask     = torch.ones(B, T, S, dtype=torch.bool, device=lat_vec.device)
        coords   = coords_tuv[:, 0, :, :2]                                     # (B,N,2)
        merged_phi = (original_phi
                      if original_phi is not None
                      else torch.ones(B, S, device=lat_vec.device))

        return lat_vec, mask, coords, merged_phi

# ================================================================
# Temporal Decoder using linear transformer
# ================================================================
def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0                     # φ(x) = ELU(x)+1  (positive)

# Causal Linear Multi-Head Attention (parallel + recurrent)
class MultiheadLinearAttention(nn.Module):
    """
    Causal linear attention that can run
        - in parallel over the whole sequence   (forward_parallel)
        - token-by-token for generation         (step)
    
    Notation  (Katharopoulos et al., 2020):
        y_t = (φ(q_t)^T  S_t) / (φ(q_t)^T  Z_t)   ,
        S_t = Σ_{τ ≤ t} φ(k_τ) ⊗ v_τ
        Z_t = Σ_{τ ≤ t} φ(k_τ)
    """
    def __init__(self, d_model: int, n_heads: int = 4,
                 feature_map=elu_feature_map, eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hdim    = d_model // n_heads
        self.eps     = eps
        self.phi     = feature_map

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # --- recurrent state for incremental decoding ---
        self.register_buffer("S", torch.zeros(1, n_heads, self.hdim, self.hdim))
        self.register_buffer("Z", torch.zeros(1, n_heads, self.hdim))

    # ------------------------------------------------------------------
    # Full-sequence parallel forward (teacher forcing during training)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)   - entire window, causal (autoregressive).
        returns
            y : (B, T, d_model)
        complexity O(B·T·h·d)  memory O(B·h·d)        (no L² term).
        """
        B, T, _ = x.shape

        # Projections -> (B, T, h, d_h)
        q = self.phi(self.W_q(x)).view(B, T, self.n_heads, self.hdim)
        k = self.phi(self.W_k(x)).view(B, T, self.n_heads, self.hdim)
        v =                self.W_v(x) .view(B, T, self.n_heads, self.hdim)

        # Prefix sums ---------------------------------------------------
        #   Z_t = Σ k_τ
        Z = torch.cumsum(k, dim=1)                                      # (B,T,h,d)

        #   S_t = Σ k_τ ⊗ v_τ        (outer product)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)                          # (B,T,h,d,d)
        S  = torch.cumsum(kv, dim=1)                                    # (B,T,h,d,d)

        # Attention -----------------------------------------------------
        # numerator  : q · S_t                -> (B,T,h,d)
        # denominator: q · Z_t (scalar/head) -> (B,T,h,1)
        num = torch.matmul(q.unsqueeze(-2), S).squeeze(-2)              # (B,T,h,d)
        den = (q * Z).sum(-1, keepdim=True) + self.eps                  # (B,T,h,1)

        y = (num / den).contiguous().view(B, T, self.d_model)           # concat heads
        return self.out(y)                                              # output proj.

    # ------------------------------------------------------------------
    # Token-by-token step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset_state(self, batch_size: int, device=None):
        self.S.zero_(); self.Z.zero_()
        if self.S.size(0) != batch_size:
            self.S = self.S.expand(batch_size, -1, -1, -1).clone()
            self.Z = self.Z.expand(batch_size, -1, -1).clone()
        if device is not None:
            self.S = self.S.to(device); self.Z = self.Z.to(device)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        B = x_t.size(0)
        if self.S.size(0) != B:              # keep batch in sync
            self.reset_state(B, x_t.device)

        q = self.phi(self.W_q(x_t)).view(B, self.n_heads, self.hdim)
        k = self.phi(self.W_k(x_t)).view(B, self.n_heads, self.hdim)
        v =                self.W_v(x_t) .view(B, self.n_heads, self.hdim)

        # 1. previous state is treated as a constant for this step
        S_prev = self.S.detach()
        Z_prev = self.Z.detach()

        # 2. candidate update for the current token
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)        # (B,h,d,d)
        S_new = S_prev + kv
        Z_new = Z_prev + k

        # 3. compute attention with the *updated* prefix statistics
        num = torch.matmul(q.unsqueeze(-2), S_new).squeeze(-2)  # (B,h,d)
        den = (q * Z_new).sum(-1, keepdim=True) + self.eps      # (B,h,1)
        out = (num / den).reshape(B, self.d_model)              # concat heads
        out = self.out(out)

        # 4. save the new statistics for the next step (no grad needed)
        self.S = S_new.detach()
        self.Z = Z_new.detach()
        return out

class TemporalDecoderLinear(nn.Module):
    """
    Forecasts a latent trajectory of length N_fore from an observation window.
    - Uses a stack of causal linear-attention blocks
    - 2-stage Heun (RK2) integrator
    - Gradient checkpointing for memory efficiency
    - Optional learnable scaling of the physical dt  (defaults to 1.0)
    """
    def __init__(self, d_model: int,
                 n_layers: int = 4,
                 n_heads : int = 4,
                 dt: float = 0.02,
                 learnable_dt: bool = True,
                 dropout: float = 0.0,
                 checkpoint_every_layer: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiheadLinearAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.pre_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model)
            ) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, d_model)

        # ---- dt -------------------------------------------------------
        self.register_buffer("dt_const", torch.tensor(dt))
        if learnable_dt:
            self.dt_scale = nn.Parameter(torch.zeros(()))  # Softplus scaling
        else:
            self.dt_scale = None

        self.use_ckpt = checkpoint_every_layer

    def _block(self, idx: int, x: torch.Tensor,
               *, incremental: bool = False) -> torch.Tensor:
        ln_pre, attn, ln_post, ffn = \
            self.pre_norms[idx], self.layers[idx], self.post_norms[idx], self.ffns[idx]

        residual = x
        x = ln_pre(x)
        x = attn.step(x) if incremental else attn(x)
        x = ln_post(residual + x)
        x = x + ffn(x)
        return x

    # gradient-checkpoint wrapper
    def _block_ckpt(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        if not (self.use_ckpt and self.training):
            return self._block(idx, x, incremental=False)

        def fn(y):
            return self._block(idx, y, incremental=False)
        return cp.checkpoint(fn, x, use_reentrant=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # parallel pass
        for idx in range(len(self.layers)):
            x_seq = self._block_ckpt(idx, x_seq)

        k1 = self.head(x_seq)
        dt = self._effective_dt()
        k2 = self.head(self._forward_no_ckpt(x_seq + dt * k1))

        return x_seq + 0.5 * dt * (k1 + k2)

    # =========================================================
    #  Autoregressive roll-out that keeps gradients (training)
    # =========================================================
    def rollout_with_grad(
        self,
        obs_window      : torch.Tensor,          # (B, T_obs, D)
        N_fore          : int,
        *,
        truncate_k      : int | None = 64,
        teacher_force_seq : torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ) -> torch.Tensor:

        assert self.training, "Call only in training mode"

        B, T_obs, D = obs_window.shape
        device      = obs_window.device

        # reset running state in every attention layer
        for lyr in self.layers:
            lyr.reset_state(B, device)

        # ------------------------------------------------------
        # 1) prime prefix statistics with the observed window
        # ------------------------------------------------------
        outputs = []
        for t in range(T_obs):
            token = obs_window[:, t]                           # (B, D)
            outputs.append(token)
            _ = self._step_layers(token)                       # incremental=True

        # ------------------------------------------------------
        # 2) autoregressive prediction
        # ------------------------------------------------------
        dt_eff       = self._effective_dt()
        steps_to_go  = N_fore - T_obs
        latent_cur   = obs_window[:, -1]                       # last observed

        for t in range(steps_to_go):

            y   = self._step_layers(latent_cur)                # context at t
            k1  = self.head(y)
            k2  = self.head(self._step_layers(latent_cur + dt_eff * k1))
            latent_next = latent_cur + 0.5 * dt_eff * (k1 + k2)

            outputs.append(latent_next)

            # ---------- teacher forcing ----------------------
            use_tf = (
                teacher_force_seq is not None
                and t < teacher_force_seq.size(1)
                and torch.rand((), device=device) < teacher_force_prob
            )
            latent_cur = teacher_force_seq[:, t] if use_tf else latent_next

            # ---------- truncated BPTT -----------------------
            if truncate_k and ((t + 1) % truncate_k == 0):
                for lyr in self.layers:
                    lyr.S = lyr.S.detach()
                    lyr.Z = lyr.Z.detach()

        return torch.stack(outputs, dim=1)                      # (B, N_fore, D)

    @torch.no_grad()
    def generate(self, obs_window: torch.Tensor, N_fore: int) -> torch.Tensor:
        B, T_obs, _ = obs_window.shape
        for l in self.layers:
            l.reset_state(B, obs_window.device)

        out = [obs_window[:, t] for t in range(T_obs)]
        latent_cur = obs_window[:, -1]

        dt = self._effective_dt()
        for _ in range(N_fore - T_obs):
            y   = self._step_layers(latent_cur)
            k1  = self.head(y)
            k2  = self.head(self._step_layers(latent_cur + dt * k1))
            latent_cur = latent_cur + 0.5 * dt * (k1 + k2)
            out.append(latent_cur)

        return torch.stack(out, 1)

    def _effective_dt(self):
        if self.dt_scale is None:
            # return self.dt_const.to(next(self.parameters()).device)
            return self.dt_const
        return self.dt_const * F.softplus(self.dt_scale)

    # shared sub-routines ------------------------------------------------
    def _forward_no_ckpt(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.layers)):
            x = self._block(idx, x, incremental=False)
        return x

    def _step_layers(self, x_t: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.layers)):
            x_t = self._block(idx, x_t, incremental=True)
        return x_t

# ================================================================
# Temporal Decoder using DelayEmbedNeuralODE
# ================================================================
class DelayODEFunc(nn.Module):
    """
    dZ/dt for the delay-embedded Neural ODE

    Parameters
    ----------
    d_model        : latent dimension of *one* token
    N_window       : size of the delay line
    hidden_dims    : list like [128, 128, 128]  (arbitrary length)
    activation     : activation for every hidden layer  (see MLP)
    norm           : None | 'layer' | 'batch'  - optional input normalisation
    """

    def __init__(
        self,
        d_model      : int,
        N_window     : int,
        hidden_dims  : list[int] | tuple[int, ...] = (128, 128),
        *,
        activation   : str = "relu",
        norm         : str | None = "layer",
    ):
        super().__init__()
        self.N_window = N_window

        # ---------- build the MLP with the user-supplied helper -------
        in_dim = N_window * d_model
        dims   = [in_dim, *hidden_dims, d_model]        # <-- [n_in, ..., n_out]
        self.mlp = MLP(dims,
                       activation        = activation,
                       final_activation  = None)        # linear output

        # ---------- optional input normalisation ----------------------
        if   norm is None:
            self.norm_in = nn.Identity()
        elif norm.lower() == "layer":
            self.norm_in = nn.LayerNorm(in_dim)
        elif norm.lower() == "batch":
            self.norm_in = nn.BatchNorm1d(in_dim)
        else:
            raise ValueError("norm must be None | 'layer' | 'batch'")

    # ------------------------------------------------------------------ 
    def forward(self, t: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Z : (B, N_window, d_model)
        returns dZ/dt with the same shape
        """
        B, Nw, D = Z.shape
        assert Nw == self.N_window, "delay length mismatch"

        dZ = torch.zeros_like(Z)
        # slots that simply age
        dZ[:, :-1] = Z[:, 1:] - Z[:, :-1]            # finite dif
        # newest token derivative predicted by the MLP
        newest_ctx = Z.reshape(B, -1)                # (B, Nw*D)
        newest_ctx = self.norm_in(newest_ctx)
        dZ[:, -1]  = self.mlp(newest_ctx)

        return dZ

class DelayEmbedNeuralODE(nn.Module):
    """
    Replaces TemporalDecoderLinear.
    .forward()      : teacher-forced next-step loss   (parallel)
    .generate()     : autoregressive roll-out         (incremental)
    .rollout_with_grad(): scheduled-sampling variant
    """
    def __init__(self,
                 d_model   : int,
                 N_window  : int,
                 hidden_dims  : list[int] | tuple[int, ...],
                 dt        : float,
                 solver    : str = "dopri5",
                 rtol:float=1e-5, atol:float=1e-6):
        super().__init__()
        self.N_window = N_window
        self.func     = DelayODEFunc(d_model, N_window, hidden_dims)
        self.solver   = solver
        self.dt       = dt
        self.rtol     = rtol
        self.atol     = atol

    # ------------ helper -------------------------------------------------
    def _ode_step(self, Z0):
        """
        Integrate from t=0 → t=dt, return Z(dt) with gradient.
        """
        t = torch.tensor([0., self.dt], device=Z0.device, dtype=Z0.dtype)
        ZT = odeint(self.func, Z0, t,
                    method=self.solver, rtol=self.rtol, atol=self.atol)
        return ZT[-1]                           # shape (B, N_w, D)

    # ------------ 1) parallel teacher-forced -----------------------------
    def forward(self, x_seq: torch.Tensor):
        """
        x_seq : (B, T_obs, D)
        returns next-step prediction (B, T_obs, D)
        """
        B, T, D = x_seq.shape
        assert T >= self.N_window

        preds = []
        # slide a causal window over the sequence
        Z = x_seq[:, :self.N_window].clone()
        for t in range(self.N_window, T):
            Z = self._ode_step(Z)
            preds.append(Z[:, -1])              # newest latent
            # teacher forcing: overwrite newest slot with ground truth
            Z[:, -1] = x_seq[:, t]

        return torch.stack(preds, 1)            # (B, T-N_window, D)

    # ------------ 2) inference / generation ------------------------------
    @torch.no_grad()
    def generate(self, obs_window: torch.Tensor, N_fore: int):
        """
        obs_window : (B, N_window, D)   -  full delay window
        N_fore     : total trajectory length to return  (≥ N_window)
        """
        Z = obs_window.clone()                 # state
        outputs = [*torch.unbind(obs_window, 1)]  # list of tensors (B,D)

        while len(outputs) < N_fore:
            Z = self._ode_step(Z)
            outputs.append(Z[:, -1])

        return torch.stack(outputs, 1)          # (B, N_fore, D)

    # ------------ 3) scheduled-sampling training path --------------------
    def rollout_with_grad(self,
                          obs_window: torch.Tensor,
                          N_fore: int,
                          *,
                          teacher_force_seq : torch.Tensor | None = None,
                          teacher_force_prob: float = 0.0,
                          truncate_k        : int = 32,
                          ):
        """
        Autoregressive roll-out that *keeps* gradients.
        """
        assert self.training
        Z = obs_window.clone()
        outputs = [*torch.unbind(obs_window, 1)]

        rng = torch.rand_like(torch.empty(1))

        steps = N_fore - obs_window.size(1)
        for k in range(steps):
            Z = self._ode_step(Z)
            newest = Z[:, -1]
            outputs.append(newest)

            # scheduled sampling
            if teacher_force_seq is not None and torch.rand(()) < teacher_force_prob:
                Z[:, -1] = teacher_force_seq[:, k]
        return torch.stack(outputs, 1)

# ================================================================
# Full-softmax temporal decoder 
# ================================================================
class MultiheadSoftmaxAttention(nn.Module):
    """
    Causal multi-head soft-max attention that always calls
    torch.nn.functional.scaled_dot_product_attention (SDPA).

    · forward(x) : B × T × D  →  B × T × D
    · step(x_t)  : B ×   D    →  B ×   D         (recurrent KV cache)
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hdim    = d_model // n_heads
        self.dropout_p = dropout

        # projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # caches for autoregressive generation
        self.K: torch.Tensor | None = None   # (B,h,T,d)
        self.V: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _split(self, x: torch.Tensor):
        """(B,T,D) -> (B,h,T,d)"""
        B, T, _ = x.shape
        return (
            x.view(B, T, self.n_heads, self.hdim)
            .transpose(1, 2)                       # B h T d
        )

    def _merge(self, x: torch.Tensor):
        """(B,h,T,d) -> (B,T,D)"""
        B, h, T, d = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, T, h * d)
        )

    # ------------------------------------------------------------------
    # full sequence (teacher forcing)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self._split(self.W_q(x))                # B h T d
        k = self._split(self.W_k(x))
        v = self._split(self.W_v(x))

        # SDPA automatically picks Flash / Triton / math
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )                                           # B h T d

        return self.out(self._merge(y))             # B T D

    # ------------------------------------------------------------------
    # incremental step
    # ------------------------------------------------------------------
    def reset_state(self, batch_size: int, device=None):
        self.K = None
        self.V = None
        self._bs = batch_size
        self._device = device

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t : (B, D)
        """
        B, _ = x_t.shape
        if (self.K is None) or (B != self._bs):
            self.reset_state(B, x_t.device)

        q = self.W_q(x_t).view(B, self.n_heads, 1, self.hdim)      # B h 1 d
        k = self.W_k(x_t).view(B, self.n_heads, 1, self.hdim)
        v = self.W_v(x_t).view(B, self.n_heads, 1, self.hdim)

        # append to cache along seq-length dimension
        self.K = k if self.K is None else torch.cat((self.K, k), dim=2)
        self.V = v if self.V is None else torch.cat((self.V, v), dim=2)

        y = F.scaled_dot_product_attention(
            q, self.K, self.V,
            attn_mask=None,
            dropout_p=0.0,            # no dropout during generation
            is_causal=True,
        )                             # B h 1 d
        y = y.squeeze(2)              # B h d
        y = y.transpose(1, 2).reshape(B, self.d_model)
        return self.out(y)

class TemporalDecoderSoftmax(nn.Module):

    def __init__(
        self,
        d_model      : int,
        n_layers     : int = 4,
        n_heads      : int = 4,
        max_len      : int = 4096,
        dt           : float = 0.02,
        learnable_dt : bool  = False,
        dropout      : float = 0.0,
        rope_base     : float = 1000.0,
        checkpoint_every_layer: bool = True,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        self.n_heads  = n_heads
        self.rope_base = rope_base
        self.layers = nn.ModuleList(
            [MultiheadSoftmaxAttention(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.pre_norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
            ) for _ in range(n_layers)
        ])

        # Heun integrator projection head
        self.head = nn.Linear(d_model, d_model)

        # learnable absolute positional embedding
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # dt handling ---------------------------------------------------
        self.register_buffer("dt_const", torch.tensor(dt))
        if learnable_dt:
            self.dt_scale = nn.Parameter(torch.zeros(()))
        else:
            self.dt_scale = None

        self.use_ckpt = checkpoint_every_layer

    # -----------------------------------------------------------------
    def apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
          x   : (B, T, D) or (B, 1, D)
          pos : (1, T, 1)  integer time-indices
        """
        B, T, D = x.shape
        d = D // 2
        # compute the per‐dim frequencies
        freq = self.rope_base ** (-torch.arange(d, device=x.device) / d) # (d,)

        angle = pos * freq.view(1,1,d)                                   # (1,T,d)
        s, c = angle.sin(), angle.cos()                                  # (1,T,d)
        x1, x2 = x[..., :d], x[..., d:]                                  # (B,T,d) each

        xr = torch.cat([
            x1 * c - x2 * s,
            x1 * s + x2 * c
        ], dim=-1)
        return xr

    def _add_pos(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if x.dim() == 3:
            B, T, D = x.shape
            # build a (1,T,1) of [offset, offset+1, …]
            pos = torch.arange(offset, offset+T, device=x.device).view(1, T, 1)
            return self.apply_rope(x, pos)
        else:
            # single token  x=(B,D) -> make it (B,1,D) + rope + squeeze
            pos = torch.tensor([[offset]], device=x.device)  # (1,1)
            return self.apply_rope(x.unsqueeze(1), pos).squeeze(1)

    def _effective_dt(self):
        if self.dt_scale is None:
            return self.dt_const
        return self.dt_const * F.softplus(self.dt_scale)

    # ------------------------------------------------------------------
    #  Single transformer block (with optional checkpointing)
    # ------------------------------------------------------------------
    def _block(self, idx: int, x: torch.Tensor, *, incremental: bool) -> torch.Tensor:
        ln_pre, attn, ln_post, ffn = \
            self.pre_norms[idx], self.layers[idx], self.post_norms[idx], self.ffns[idx]

        residual = x
        x = ln_pre(x)
        x = attn.step(x) if incremental else attn(x)
        x = ln_post(residual + x)
        x = x + ffn(x)
        return x

    def _block_ckpt(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        if not (self.use_ckpt and self.training):
            return self._block(idx, x, incremental=False)

        def fn(y):
            return self._block(idx, y, incremental=False)
        return cp.checkpoint(fn, x, use_reentrant=False)

    # ------------------------------------------------------------------
    #  Parallel forward (teacher forcing over full sequence)
    # ------------------------------------------------------------------
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq : (B, T, D)
        """
        x_seq = self._add_pos(x_seq, 0)

        for idx in range(len(self.layers)):
            x_seq = self._block_ckpt(idx, x_seq)

        k1 = self.head(x_seq)
        dt = self._effective_dt()
        k2 = self.head(self._forward_no_ckpt(x_seq + dt * k1))
        return x_seq + 0.5 * dt * (k1 + k2)

    # ------------------------------------------------------------------
    #  Autoregressive rollout with gradients (training)
    # ------------------------------------------------------------------
    def rollout_with_grad(
        self,
        obs_window      : torch.Tensor,   # (B, T_obs, D)
        N_fore          : int,
        *,
        truncate_k      : int | None = 64,
        teacher_force_seq : torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ) -> torch.Tensor:

        assert self.training, "Call only in training mode"
        B, T_obs, D = obs_window.shape
        dev = obs_window.device

        # reset state in every attention layer
        for l in self.layers:
            l.reset_state(B, dev)

        # ----------------------------------
        # 1. prime prefix with observation
        # ----------------------------------
        outputs = []
        for t in range(T_obs):
            token_raw = obs_window[:, t]
            token_pe = self._add_pos(token_raw, t)          # (B,D)
            outputs.append(token_raw)
            _ = self._step_layers(token_pe)                        # caches grow

        # ----------------------------------
        # 2. autoregressive prediction
        # ----------------------------------
        dt_eff     = self._effective_dt()
        steps_left = N_fore - T_obs
        latent_cur = obs_window[:, -1]                          # last GT state

        for step in range(steps_left):
            pos_idx = T_obs + step

            y   = self._step_layers(self._add_pos(latent_cur, pos_idx))
            k1  = self.head(y)
            k2  = self.head(self._step_layers(
                     self._add_pos(latent_cur + dt_eff * k1, pos_idx)))
            latent_next = latent_cur + 0.5 * dt_eff * (k1 + k2)

            outputs.append(latent_next)

            # teacher forcing
            use_tf = (
                teacher_force_seq is not None
                and step < teacher_force_seq.size(1)
                and torch.rand((), device=dev) < teacher_force_prob
            )
            latent_cur = teacher_force_seq[:, step] if use_tf else latent_next

            # truncated BPTT
            if truncate_k and ((step + 1) % truncate_k == 0):
                for l in self.layers:
                    if l.K is not None:
                        l.K = l.K.detach()
                        l.V = l.V.detach()

        return torch.stack(outputs, 1)     # (B, N_fore, D)

    # ------------------------------------------------------------------
    #  Greedy generation (no grad) – evaluation / inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, obs_window: torch.Tensor, N_fore: int) -> torch.Tensor:
        """
        obs_window : (B, T_obs, D)
        returns     : (B, N_fore, D)   first T_obs tokens are the inputs
        """
        B, T_obs, _ = obs_window.shape
        dev = obs_window.device
        for l in self.layers:
            l.reset_state(B, dev)

        out = []
        for t in range(T_obs):
            token_raw = obs_window[:, t]
            token_pe  = self._add_pos(token_raw, t)
            out.append(token_raw)
            _ = self._step_layers(token_pe)

        dt = self._effective_dt()
        latent_cur = obs_window[:, -1]

        for step in range(N_fore - T_obs):
            pos_idx = T_obs + step
            y   = self._step_layers(self._add_pos(latent_cur, pos_idx))
            k1  = self.head(y)
            k2  = self.head(self._step_layers(
                     self._add_pos(latent_cur + dt * k1, pos_idx)))
            latent_cur = latent_cur + 0.5 * dt * (k1 + k2)
            out.append(latent_cur)

        return torch.stack(out, 1)

    # ------------------------------------------------------------------
    #  shared utilities
    # ------------------------------------------------------------------
    def _forward_no_ckpt(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.layers)):
            x = self._block(idx, x, incremental=False)
        return x

    def _step_layers(self, x_t: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.layers)):
            x_t = self._block(idx, x_t, incremental=True)
        return x_t

# ================================================================
# Uncertainty-aware softmax temporal decoder 
# ================================================================
class ud_MultiheadSoftmaxAttention(nn.Module):
    def __init__(self, d_model, num_heads, gamma=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable weights for penalties (minimal addition)
        self.w_sigma = nn.Parameter(torch.tensor(1.0))
        self.w_phi = nn.Parameter(torch.tensor(1.0))
        self.gamma = gamma

        # Initialize instance vars for step method
        self.K = None
        self.V = None

    def forward(self, query, key, value, phi=None, logvar=None):
        B, T, _ = query.shape
        assert query.shape == key.shape == value.shape, "Query, key, value shapes must match"
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # New: Uncertainty-aware modulation (penalize high logvar/low phi)
        if logvar is not None:
            assert logvar.shape[:2] == (B, T), f"Logvar shape mismatch: expected (B, T, *), got {logvar.shape}"
            logvar_broadcast = logvar.unsqueeze(1).expand(-1, self.num_heads, T, -1).mean(dim=-1, keepdim=True)  # Avg per time-step, expanded for heads
            scores -= self.w_sigma * logvar_broadcast.unsqueeze(2)  # Broadcast to match scores shape
        if phi is not None:
            assert phi.shape[:2] == (B, T), f"Phi shape mismatch: expected (B, T, *), got {phi.shape}"
            phi_broadcast = phi.unsqueeze(1).expand(-1, self.num_heads, T, -1).mean(dim=-1, keepdim=True)
            scores -= self.w_phi * (1 - phi_broadcast.unsqueeze(2))
            # Additional phi modulation from my proposal
            phi_denom = torch.clamp(phi_broadcast.unsqueeze(2) ** self.gamma, min=1e-6)  # Clamp for stability
            scores = scores / phi_denom
        
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    def reset_state(self, B, device):
        self.K = self.V = None

    def step(self, x_t, phi=None, logvar=None):
        # x_t : (B,1,D)
        assert x_t.dim() == 3 and x_t.shape[1] == 1, f"Expected (B,1,D), got {x_t.shape}"
        q = self.q_proj(x_t).view(x_t.shape[0], 1, self.num_heads, self.head_dim).transpose(1, 2)  # Multi-head reshape
        k = self.k_proj(x_t).view(x_t.shape[0], 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_t).view(x_t.shape[0], 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_flat = k.transpose(1, 2).squeeze(1)  # Flatten for cat
        v_flat = v.transpose(1, 2).squeeze(1)
        self.K = k_flat if self.K is None else torch.cat([self.K, k_flat], 1)
        self.V = v_flat if self.V is None else torch.cat([self.V, v_flat], 1)

        scores = (q @ self.K.view(q.shape[0], self.num_heads, -1, self.head_dim).transpose(-2,-1)) / math.sqrt(self.head_dim)
        if logvar is not None:
            scores -= self.w_sigma * logvar.mean(dim=-1, keepdim=True).unsqueeze(1)  # Expanded for heads
        if phi is not None:
            scores -= self.w_phi * (1 - phi.mean(dim=-1, keepdim=True).unsqueeze(1))
            phi_denom = torch.clamp(phi.mean(dim=-1, keepdim=True).unsqueeze(1) ** self.gamma, min=1e-6)
            scores = scores / phi_denom

        attn = scores.softmax(-1)
        out  = attn @ self.V.view(attn.shape[0], self.num_heads, -1, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(x_t.shape[0], 1, self.d_model)
        return self.out_proj(out)

class UncertaintyAwareTemporalDecoder(TemporalDecoderSoftmax):
    def __init__(self, *args, unc_token_dim=16, gamma=1.0, **kw):
        super().__init__(*args, **kw)  

        self.layers = nn.ModuleList(
            [ud_MultiheadSoftmaxAttention(self.d_model, self.n_heads, gamma=gamma) for _ in range(self.n_layers)]
        )
        self.var_head  = nn.Linear(self.d_model, self.d_model)
        self.unc_proj  = nn.Linear(unc_token_dim, self.d_model)
        self.w_sigma   = nn.Parameter(torch.tensor(1.0))
        self.w_phi     = nn.Parameter(torch.tensor(1.0))
        self.gamma     = gamma

    def _augment_latents(self, z, logvar, phi, stats):
        B,T,D = z.shape
        assert z.shape[-1] == self.d_model, f"Latent dim mismatch: {z.shape[-1]} != {self.d_model}"
        # pool & project
        log_sigma = logvar.mean((-1,)) if logvar is not None else torch.zeros((B,T), device=z.device)  # (B,T)
        phi_mean  = phi.mean(-1).unsqueeze(1).expand(-1,T) if phi is not None else torch.ones((B,T), device=z.device)  # (B,T)
        
        s    = torch.zeros_like(log_sigma) if stats is None else stats.mean(-1)
        u    = torch.stack([log_sigma, phi_mean, s], -1)    

        # Project directly to d_model instead of cat/truncate
        u_proj = self.unc_proj(u)  
        # Additive augmentation for efficiency
        return z + u_proj  

    def _heun_step(self, latent_cur, dt, logvar_cur=None):  # Modified for adaptive integration
        y = self.head(latent_cur)
        k1 = y
        y_mid = latent_cur + dt * k1
        k2 = self.head(y_mid)
        
        # New: Predict logvar (conditioned on current latent, like decoder)
        logvar_pred = self.var_head(y_mid)
        
        # Adaptive scaling (downweight by normalized variance)
        var_norm = torch.tanh(torch.exp(logvar_pred).mean(-1,keepdim=True))   # (B,1), bounded [0,1]
        update = 0.5 * dt * (k1 + k2) * (1 - var_norm)
        
        latent_next = latent_cur + update
        return latent_next, logvar_pred

    def forward(self, x_seq: torch.Tensor, phi=None, logvar=None, stats=None):  # Modified to match original signature + optional uncertainty
        augmented = self._augment_latents(x_seq, logvar, phi, stats) if phi is not None or logvar is not None or stats is not None else x_seq
        # Call super's forward for the core logic (Heun integration over full sequence)
        mean_out = super().forward(augmented)
        logvar_out = self.var_head(mean_out) if self.training else None  # Predict logvar only if needed
        return mean_out, logvar_out  # Return tuple for compatibility (adapter handles it)

    @torch.no_grad()
    def generate(self, obs_window: torch.Tensor, N_fore: int, phi=None, initial_logvar=None, stats=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Matches original signature: primes with obs_window, generates N_fore steps.
        Optional uncertainty params for modulation and logvar propagation.
        Returns (traj, traj_logvar) or (traj, None) if no initial_logvar.
        """
        B, T_obs, _ = obs_window.shape
        dev = obs_window.device
        for l in self.layers:
            l.reset_state(B, dev)

        out = []
        out_logvar = [] if initial_logvar is not None else None
        cur_logvar = initial_logvar  # Assume for last token of obs_window

        # 1. Prime prefix with observation (build K/V state, with optional uncertainty)
        for t in range(T_obs):
            token_raw = obs_window[:, t : t+1]  # (B,1,D)
            token_pe = self._add_pos(token_raw.squeeze(1), t).unsqueeze(1)  # (B,1,D)
            _ = self._step_layers(token_pe, phi=phi[:, t:t+1] if phi is not None else None, logvar=cur_logvar)  # Step with uncertainty
            out.append(token_raw.squeeze(1))
            if out_logvar is not None:
                out_logvar.append(cur_logvar)

        # Last latent from priming
        latent_cur = obs_window[:, -1 :]

        # 2. Autoregressive prediction using Heun with logvar propagation
        dt = self._effective_dt()
        for step in range(N_fore - T_obs):
            pos_idx = T_obs + step
            # Add pos and step through layers with uncertainty
            token_pe = self._add_pos(latent_cur, pos_idx)
            y = self._step_layers(token_pe, phi=phi[:, pos_idx:pos_idx+1] if phi is not None else None, logvar=cur_logvar)
            latent_next, next_logvar = self._heun_step(y, dt, cur_logvar)
            out.append(latent_next.squeeze(1))
            if out_logvar is not None:
                out_logvar.append(next_logvar)
            latent_cur = latent_next
            cur_logvar = next_logvar

        traj = torch.stack(out, 1)  # (B, N_fore, D)
        traj_logvar = torch.stack(out_logvar, 1) if out_logvar is not None else None
        return traj, traj_logvar

    def rollout_with_grad(
        self,
        obs_window      : torch.Tensor,   # (B, T_obs, D)
        N_fore          : int,
        *,
        truncate_k      : int | None = 64,
        teacher_force_seq : torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
        phi=None, initial_logvar=None, stats=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Matches original signature: primes with obs_window, supports truncate_k and teacher forcing.
        Optional uncertainty params for modulation and logvar propagation.
        Returns (traj, traj_logvar) or (traj, None) if no initial_logvar.
        """
        assert self.training, "Call only in training mode"
        B, T_obs, D = obs_window.shape
        dev = obs_window.device

        # reset state in every attention layer
        for l in self.layers:
            l.reset_state(B, dev)

        # ----------------------------------
        # 1. prime prefix with observation
        # ----------------------------------
        outputs = []
        outputs_logvar = [] if initial_logvar is not None else None
        cur_logvar = initial_logvar  # Assume for last of obs_window
        for t in range(T_obs):
            token_raw = obs_window[:, t : t+1]  # (B,1,D)
            token_pe = self._add_pos(token_raw.squeeze(1), t).unsqueeze(1)
            _ = self._step_layers(token_pe, phi=phi[:, t:t+1] if phi is not None else None, logvar=cur_logvar)  # caches grow
            outputs.append(token_raw.squeeze(1))
            if outputs_logvar is not None:
                outputs_logvar.append(cur_logvar)

        # ----------------------------------
        # 2. autoregressive prediction
        # ----------------------------------
        dt_eff     = self._effective_dt()
        steps_left = N_fore - T_obs
        latent_cur = obs_window[:, -1 : ]  # (B,1,D)

        for step in range(steps_left):
            pos_idx = T_obs + step

            y = self._step_layers(self._add_pos(latent_cur.squeeze(1), pos_idx).unsqueeze(1), phi=phi[:, pos_idx:pos_idx+1] if phi is not None else None, logvar=cur_logvar)
            latent_next, next_logvar = self._heun_step(y, dt_eff, cur_logvar)

            outputs.append(latent_next.squeeze(1))
            if outputs_logvar is not None:
                outputs_logvar.append(next_logvar)

            # teacher forcing
            use_tf = (
                teacher_force_seq is not None
                and step < teacher_force_seq.size(1)
                and torch.rand((), device=dev) < teacher_force_prob
            )
            latent_cur = teacher_force_seq[:, step : step+1] if use_tf else latent_next
            cur_logvar = initial_logvar if use_tf else next_logvar  # Reset logvar if teacher forcing (assume GT has no var)

            # truncated BPTT
            if truncate_k and ((step + 1) % truncate_k == 0):
                for l in self.layers:
                    if l.K is not None:
                        l.K = l.K.detach()
                        l.V = l.V.detach()

        traj = torch.stack(outputs, 1)  # (B, N_fore, D)
        traj_logvar = torch.stack(outputs_logvar, 1) if outputs_logvar is not None else None
        return traj, traj_logvar

    # ------------------------------------------------------------------
    #  shared utilities (overridden to pass uncertainty to attn.step/forward)
    # ------------------------------------------------------------------
    def _block(self, idx: int, x: torch.Tensor, *, incremental: bool, phi=None, logvar=None) -> torch.Tensor:
        ln_pre, attn, ln_post, ffn = \
            self.pre_norms[idx], self.layers[idx], self.post_norms[idx], self.ffns[idx]

        residual = x
        x = ln_pre(x)
        if incremental:
            x = attn.step(x, phi=phi, logvar=logvar)  # Pass to step
        else:
            x = attn(x, x, x, phi=phi, logvar=logvar)  # Self-attn with uncertainty
        x = ln_post(residual + x)
        x = x + ffn(x)
        return x

    def _step_layers(self, x_t: torch.Tensor, phi=None, logvar=None) -> torch.Tensor:
        for idx in range(len(self.layers)):
            x_t = self._block(idx, x_t, incremental=True, phi=phi, logvar=logvar)
        return x_t

    def _forward_no_ckpt(self, x: torch.Tensor, phi=None, logvar=None, stats=None) -> torch.Tensor:
        augmented = self._augment_latents(x, logvar, phi, stats) if phi is not None or logvar is not None or stats is not None else x
        for idx in range(len(self.layers)):
            augmented = self._block(idx, augmented, incremental=False, phi=phi, logvar=logvar)
        return augmented

class CAU_MultiheadCrossAttention(nn.Module):
    """
    Causal multi-head cross-attention with optional importance weights.

    query:      [B, T_q, D] or [B, 1, N_q, D] or [B, D]
    key_value:  [B, T_kv, D] or [B, T_kv, N_s, D] or [B, N_s, D] or [B, D]

    Importance weights (imp_weights):
      - If key_value provides a sensor axis of length N_s, pass imp_weights as [B, N_s].
      - We append these per-step to an internal importance cache aligned with K/V tokens.
      - Applied as additive mask: log(clamp(imp, eps, 1.0)) added to attention logits.

    Caches:
      - self.K, self.V: [B, h, S, d], where S is the accumulated source length.
      - self.imp_cache: [B, S] for per-source importance, optional.
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hdim = d_model // n_heads
        self.dropout_p = dropout

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # K/V caches (already split into heads) and aligned importance cache
        self.K: torch.Tensor | None = None  # [B, h, S, d]
        self.V: torch.Tensor | None = None  # [B, h, S, d]
        self.imp_cache: torch.Tensor | None = None  # [B, S]
        self._bs: int | None = None
        self._device = None

    def _split(self, x: torch.Tensor):
        # x: [B, T, D] -> [B, h, T, d]
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.hdim).transpose(1, 2)

    def _merge(self, x: torch.Tensor):
        # x: [B, h, T, d] -> [B, T, D]
        B, h, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d)

    def reset_state(self, batch_size: int, device=None):
        self.K = None
        self.V = None
        self.imp_cache = None
        self._bs = batch_size
        self._device = device

    @staticmethod
    def _to_B_T_D(x: torch.Tensor):
        """
        Normalize x to [B, T, D] and return a callable to restore original shape.
        Supported:
          - [B, D] -> [B, 1, D]
          - [B, T, D] -> [B, T, D]
          - [B, T, N, D] -> [B, T*N, D]
        """
        if x.dim() == 2:
            # [B, D]
            B, D = x.shape
            xf = x.unsqueeze(1)  # [B, 1, D]
            def restore(y):  # y: [B, 1, D]
                return y.squeeze(1)  # [B, D]
            return xf, restore

        if x.dim() == 3:
            # [B, T, D]
            B, T, D = x.shape
            def restore(y):  # [B, T, D]
                return y
            return x, restore

        if x.dim() == 4:
            # [B, T, N, D] -> flatten to [B, T*N, D]
            B, T, N, D = x.shape
            xf = x.reshape(B, T * N, D)
            def restore(y):  # [B, T*N, D] -> [B, T, N, D]
                return y.view(B, T, N, D)
            return xf, restore

        raise ValueError(f"Unsupported tensor rank: {x.shape}")

    def _append_imp(self, imp_new: torch.Tensor | None, Tk: int, *, B: int, device, dtype):
        """
        Append per-source importance for the newly appended KV tokens (length Tk).
        imp_new: [B, Tk] or None -> appends ones if None.
        """
        if imp_new is None:
            imp_new = torch.ones(B, Tk, device=device, dtype=dtype)
        if self.imp_cache is None:
            self.imp_cache = imp_new
        else:
            self.imp_cache = torch.cat([self.imp_cache, imp_new], dim=1)  # [B, S+Tk]

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, imp_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parallel attention (no caching).
        query:     [B, T_q, D] or [B, T_q, N_q, D] or [B, D]
        key_value: [B, T_kv, D] or [B, T_kv, N_s, D] or [B, N_s, D] or [B, D]
        imp_weights:
          - If key_value provides N_s, pass [B, N_s]; repeated over time if T_kv > 1.
        """
        q, restore_q = self._to_B_T_D(query)      # [B, T_q, D]
        kv, _ = self._to_B_T_D(key_value)         # [B, T_kv', D] (T_kv' may be T_kv or T_kv*N_s)

        B, T_q, _ = q.shape
        B2, T_kv_flat, D = kv.shape
        assert B == B2

        qh = self._split(self.W_q(q))             # [B, h, T_q, d]
        kh = self._split(self.W_k(kv))            # [B, h, T_kv_flat, d]
        vh = self._split(self.W_v(kv))            # [B, h, T_kv_flat, d]

        # Build importance mask aligned with source tokens (T_kv_flat)
        attn_mask = None
        if imp_weights is not None:
            if key_value.dim() == 4:  # [B, T_kv, N_s, D] flattened -> [B, T_kv*N_s, D]
                Bk, T_kv, N_s, _ = key_value.shape
                assert Bk == B
                assert imp_weights.shape == (B, N_s), f"imp_weights must be [B, N_s]=[{B}, {N_s}], got {imp_weights.shape}"
                imp_full = imp_weights.unsqueeze(1).expand(B, T_kv, N_s).reshape(B, T_kv * N_s)  # [B, T_kv*N_s]
            elif key_value.dim() == 3:
                # If user provided [B, T_kv] we can use it; otherwise ignore
                if imp_weights.shape == (B, key_value.shape[1]):
                    imp_full = imp_weights  # [B, T_kv]
                else:
                    imp_full = None
            else:
                imp_full = None

            if imp_full is not None:
                log_imp = torch.log(torch.clamp(imp_full, min=1e-6)).to(qh.dtype)  # [B, S]
                attn_mask = log_imp.view(B, 1, 1, -1)  # [B, 1, 1, S], broadcast over heads and T_q

        y = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True  # causal along the flattened time-major source axis
        )  # [B, h, T_q, d]

        out = self.out(self._merge(y))  # [B, T_q, D]
        return restore_q(out)

    def step(self, query_t: torch.Tensor, key_value_t: torch.Tensor, imp_weights_t: torch.Tensor | None = None, append: bool = True) -> torch.Tensor:
        """
        Incremental attention with caching.

        query_t:
          - [B, D] or [B, 1, D] or [B, 1, N_q, D] or [B, N_q, D]  (returns matching shape)
        key_value_t:
          - [B, D] or [B, 1, D] or [B, N_s, D] or [B, 1, N_s, D]

        imp_weights_t:
          - If key_value_t provides N_s (i.e., has a sensor axis), pass [B, N_s].
            These are appended to an internal importance cache aligned with K/V tokens.
          - If kv has no sensor axis, importance is ignored (treated as ones).

        append: If False, compute with temporary KV/imp (no permanent append) for efficiency in Heun midpoint.
        """
        q, restore_q = self._to_B_T_D(query_t)    # [B, T_q, D]
        kv, _ = self._to_B_T_D(key_value_t)       # [B, T_kv', D] where T_kv' could be N_s (for sensors-at-step) or 1 (for CLS-at-step)

        B, T_q, _ = q.shape
        B2, Tk, D = kv.shape
        assert B == B2

        # Project and split
        qh = self._split(self.W_q(q))             # [B, h, T_q, d]
        k_new = self._split(self.W_k(kv))         # [B, h, Tk, d]
        v_new = self._split(self.W_v(kv))         # [B, h, Tk, d]

        # Importance for this step: detect if kv provided a sensor axis
        imp_step = None
        if imp_weights_t is not None:
            # We consider kv had a sensor axis if imp_weights_t.dim() == 2 and shapes match
            if imp_weights_t.dim() == 2 and imp_weights_t.shape[0] == B and imp_weights_t.shape[1] == Tk:
                imp_step = imp_weights_t.to(q.dtype)
        # If None, treat as ones (will be handled below)

        if append:
            # Append to caches
            if self.K is None:
                self.K = k_new
                self.V = v_new
            else:
                self.K = torch.cat([self.K, k_new], dim=2)  # concat along source length S
                self.V = torch.cat([self.V, v_new], dim=2)
            self._append_imp(imp_step, Tk=Tk, B=B, device=q.device, dtype=q.dtype)
            K_use = self.K
            V_use = self.V
            imp_use = self.imp_cache
        else:
            # Temporary (no append) for efficiency
            K_use = torch.cat([self.K, k_new], dim=2) if self.K is not None else k_new
            V_use = torch.cat([self.V, v_new], dim=2) if self.V is not None else v_new
            # Mimic _append_imp for temp imp
            imp_new = imp_step if imp_step is not None else torch.ones(B, Tk, device=q.device, dtype=q.dtype)
            imp_use = torch.cat([self.imp_cache, imp_new], dim=1) if self.imp_cache is not None else imp_new

        # Build additive attention mask from (possibly temp) cached importance
        S = K_use.size(2)
        attn_mask = None
        if imp_use is not None:
            # imp_use: [B, S] aligned with K/V positions
            log_imp = torch.log(torch.clamp(imp_use, min=1e-6)).to(qh.dtype)  # [B, S]
            attn_mask = log_imp.view(B, 1, 1, S)  # [B, 1, 1, S]

        y = F.scaled_dot_product_attention(
            qh, K_use, V_use,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )  # [B, h, T_q, d]

        out = self.out(self._merge(y))  # [B, T_q, D]
        return restore_q(out)

# Main Class: TemporalDecoderHierarchical
class _TemporalDecoderHierarchical(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 4096,
        dt: float = 0.02,
        learnable_dt: bool = False,
        dropout: float = 0.0,
        rope_base: float = 1000.0,
        checkpoint_every_layer: bool = True,
        imp_threshold: float = 0.1,  # Mask if imp < threshold
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.imp_threshold = imp_threshold

        # Hierarchical layers (Stack n_layers of hierarchical blocks)
        self.cls_self_attns     = nn.ModuleList([MultiheadSoftmaxAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.cls_cross_attns    = nn.ModuleList([CAU_MultiheadCrossAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        # self.sensor_self_attns  = nn.ModuleList([MultiheadSoftmaxAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.sensor_cross_attns = nn.ModuleList([CAU_MultiheadCrossAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

        self.pre_norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])  # For CLS and sensors
        self.post_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
            ) for _ in range(n_layers * 2)  # For CLS and sensors
        ])

        # Heun heads (separate for CLS and sensors)
        self.cls_head    = nn.Linear(d_model, d_model)
        self.sensor_head = nn.Linear(d_model, d_model)

        # Learnable positional embedding
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # dt handling 
        self.register_buffer("dt_const", torch.tensor(dt))
        if learnable_dt:
            self.dt_scale = nn.Parameter(torch.zeros(()))
        else:
            self.dt_scale = None

        self.use_ckpt = checkpoint_every_layer

    def apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
        x:  [B, T, M, D]  or [B, M, D] or [B, D]
        pos: (1, T, 1) integer/float time indices
        """
        assert x.dim() in (2, 3, 4), f"apply_rope: unsupported x.dim={x.dim()}"
        orig_dim = x.dim()

        # Lift to 4D: [B, T, M, D]
        if orig_dim == 2:          # [B, D] -> [B, 1, 1, D]
            x = x.unsqueeze(1).unsqueeze(1)
        elif orig_dim == 3:        # [B, M, D] -> [B, 1, M, D]
            x = x.unsqueeze(1)
        # else: already [B, T, M, D]

        B, T, M, D = x.shape
        assert D % 2 == 0, f"apply_rope: D must be even, got D={D}"
        d = D // 2

        # Frequencies and angles
        freq = self.rope_base ** (-torch.arange(d, device=x.device, dtype=x.dtype) / d)  # [d]
        # pos: [1, T, 1] -> [1, T, 1, 1], broadcast with freq -> [1, T, 1, d]
        angle = pos.to(x.dtype).unsqueeze(-1) * freq.view(1, 1, 1, d)

        s, c = angle.sin(), angle.cos()
        x1, x2 = x[..., :d], x[..., d:]  # [B, T, M, d]
        xr = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)  # [B, T, M, D]

        # Restore original rank
        if orig_dim == 2: xr = xr.squeeze(1).squeeze(1)    # -> [B, D]
        elif orig_dim == 3: xr = xr.squeeze(1)             # -> [B, M, D]

        return xr

    def _add_pos(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Adds rotary position embeddings to x. Accepts x with shape:
        - [B, T, M, D]  (T steps)
        - [B, M, D]     (T=1)
        - [B, D]        (T=1, M=1)
        """
        if x.dim() == 4:
            T = x.shape[1]
        elif x.dim() in (2, 3):
            T = 1
        else:
            raise ValueError(f"_add_pos: unsupported x.dim={x.dim()}")
        pos = torch.arange(offset, offset + T, device=x.device, dtype=x.dtype).view(1, T, 1)  # [1, T, 1]
        return self.apply_rope(x, pos)

    def _ensure_imp(self, imp, *, B: int, N_s: int, device, dtype=torch.float32):
        if imp is None:
            return torch.ones(B, N_s, device=device, dtype=dtype)
        return imp

    def _effective_dt(self):
        if self.dt_scale is None:
            return self.dt_const
        return self.dt_const * F.softplus(self.dt_scale)

    # ------------------------------------------------------------------
    # Hierarchical Block (core layer)
    # ------------------------------------------------------------------
    def _block(
        self,
        idx: int,
        cls: torch.Tensor,
        sensors: torch.Tensor,
        imp: torch.Tensor,
        *,
        incremental: bool,
        append: bool = True,  # New: Pass to .step() for no-append mode
    ) -> tuple:
        """
        Core hierarchical layer.

        cls     : [B, T, 1, D] (parallel)  or [B, 1, D] (incremental)
        sensors : [B, T, N_s, D]           or [B, N_s, D]
        imp     : [B, N_s]
        """

        # -----------------------------------------------------------
        # 1.  Resolve shapes
        # -----------------------------------------------------------
        if cls.dim() == 4:                              # parallel mode
            B, T, _, D = cls.shape
        elif cls.dim() == 3:                            # incremental mode
            B, _, D = cls.shape
            T = 1                                       # single-step
        else:
            raise ValueError("Unexpected cls shape")

        if sensors.dim() == 4:                          # parallel
            _, _, N_s, _ = sensors.shape
        elif sensors.dim() == 3:                        # incremental
            _, N_s, _ = sensors.shape
        else:
            raise ValueError("Unexpected sensors shape")

        # -----------------------------------------------------------
        # 2.  CLS stream
        # -----------------------------------------------------------
        cls_residual = cls
        cls_normed = self.pre_norms[idx](cls)  # [B, T, 1, D] or [B, 1, D]

        # Squeeze for attention compatibility (solution for CLS without revising attention class)
        cls_squeezed = cls_normed.squeeze(2 if not incremental else 1)  # [B, T, D] or [B, D]
        if incremental:
            cls_attn = self.cls_self_attns[idx].step(cls_squeezed)
        else:
            cls_attn = self.cls_self_attns[idx](cls_squeezed)
        # Unsqueeze back
        cls_attn = cls_attn.unsqueeze(2 if not incremental else 1)  # [B, T, 1, D] or [B, 1, D]
        # cls_attn = cls_normed

        if incremental:
            cls_cross = self.cls_cross_attns[idx].step(cls_normed, sensors, imp, append=append)
        else:
            cls_cross = self.cls_cross_attns[idx](cls_normed, sensors, imp)
        cls = cls_attn + cls_cross
        
        cls = self.post_norms[idx](cls_residual + cls)
        cls = cls + self.ffns[idx](cls)

        # -----------------------------------------------------------
        # 3.  Sensor stream
        # -----------------------------------------------------------
        # LayerNorm across all sensors at each time-step
        sensors = self.pre_norms[idx + self.n_layers](sensors)  # [B, T, N_s, D] or [B, 1, N_s, D]

        # First self-attention per sensor
        if incremental:
            sensors_sa = sensors.view(B * N_s, D)                  # [B·N_s, D]
            # sensors_sa = self.sensor_self_attns[idx].step(sensors_sa)
            sensors = sensors_sa.view(B, 1, N_s, D)
        else:
            sensors_sa = sensors.view(B * N_s, T, D)               # [B·N_s, T, D]
            # sensors_sa = self.sensor_self_attns[idx](sensors_sa)
            sensors = sensors_sa.view(B, T, N_s, D)

        # Cross attention (sensor → CLS)
        if incremental:
            sensors = sensors + self.sensor_cross_attns[idx].step(sensors, cls, imp)
        else:
            sensors = sensors + self.sensor_cross_attns[idx](sensors, cls, imp)

        # Post-norm + feed-forward
        sensors = self.post_norms[idx + self.n_layers](sensors)
        sensors = sensors + self.ffns[idx + self.n_layers](sensors)

        # -----------------------------------------------------------
        # 4.  Importance masking
        # -----------------------------------------------------------
        # mask = (imp.unsqueeze(1).unsqueeze(-1) >= self.imp_threshold)  # [B, 1, N_s, 1]
        # sensors = sensors * mask.float()

        return cls, sensors

    def _block_ckpt(self, idx: int, cls: torch.Tensor, sensors: torch.Tensor, imp: torch.Tensor) -> tuple:
        if not (self.use_ckpt and self.training):
            return self._block(idx, cls, sensors, imp, incremental=False)

        def fn(c, s, i):
            return self._block(idx, c, s, i, incremental=False)
        return cp.checkpoint(fn, cls, sensors, imp, use_reentrant=False)


    def forward(self, x_seq: torch.Tensor, imp: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_seq: [B, T, 1 + N_s, D]
        imp: [B, N_s]
        """
        B, T, M, D = x_seq.shape
        N_s = M - 1

        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=x_seq.device, dtype=x_seq.dtype)

        x_seq = self._add_pos(x_seq, 0)
        cls = x_seq[:, :, :1, :]  # [B, T, 1, D]
        sensors = x_seq[:, :, 1:, :]  # [B, T, N_s, D]

        for idx in range(self.n_layers):
            cls, sensors = self._block_ckpt(idx, cls, sensors, imp)

        # Hierarchical Heun (aligns with framework: Evolve CLS first, then sensors)
        dt = self._effective_dt()
        k1_cls = self.cls_head(cls)
        k1_sensors = self.sensor_head(sensors)
        cls_mid = cls + dt * k1_cls
        sensors_mid = sensors + dt * k1_sensors
        # Pass mid through no-ckpt forward
        cls_mid, sensors_mid = self._forward_no_ckpt(cls_mid, sensors_mid, imp)
        k2_cls = self.cls_head(cls_mid)
        k2_sensors = self.sensor_head(sensors_mid)
        cls_next = cls + 0.5 * dt * (k1_cls + k2_cls)
        sensors_next = sensors + 0.5 * dt * (k1_sensors + k2_sensors)

        return torch.cat([cls_next, sensors_next], dim=2)  # [B, T, 1 + N_s, D]


    def rollout_with_grad(
        self,
        obs_window: torch.Tensor,  # [B, T_obs, 1 + N_s, D]
        N_fore: int,
        imp: torch.Tensor | None = None,  # [B, N_s]
        *,
        truncate_k: int | None = 64,
        teacher_force_seq: torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ) -> torch.Tensor:
        assert self.training, "Call only in training mode"
        B, T_obs, M, D = obs_window.shape
        N_s = M - 1
        dev = obs_window.device
        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=obs_window.device, dtype=obs_window.dtype)

        # Reset states (hierarchical caches)
        for l in range(self.n_layers):
            self.cls_self_attns[l].reset_state(B, dev)
            self.cls_cross_attns[l].reset_state(B, dev)
            # self.sensor_self_attns[l].reset_state(B * N_s, dev)  # Batched for sensors
            self.sensor_cross_attns[l].reset_state(B, dev)

        # 1. Prime prefix
        outputs = []
        for t in range(T_obs):
            token_raw = obs_window[:, t]  # [B, 1 + N_s, D]
            token_pe = self._add_pos(token_raw.unsqueeze(1), t).squeeze(1)  # [B, 1 + N_s, D]
            cls_t = token_pe[:, :1, :]
            sensors_t = token_pe[:, 1:, :]
            _ = self._step_layers(cls_t, sensors_t, imp)  # Builds caches
            outputs.append(token_raw)

        # 2. Autoregressive prediction with hierarchical Heun
        dt_eff = self._effective_dt()
        steps_left = N_fore - T_obs
        latent_cur = obs_window[:, -1]  # [B, 1 + N_s, D]
        cls_cur = latent_cur[:, :1, :]
        sensors_cur = latent_cur[:, 1:, :]

        for step in range(steps_left):
            pos_idx = T_obs + step
            cls_y, sensors_y = self._step_layers(
                self._add_pos(cls_cur.unsqueeze(1), pos_idx).squeeze(1),
                self._add_pos(sensors_cur.unsqueeze(1), pos_idx).squeeze(1),
                imp,
                append=True  # Append for k1
            )

            k1_cls = self.cls_head(cls_y)                  # [B, 1, D]
            k1_sensors = self.sensor_head(sensors_y)       # [B, 1, N_s, D]
            if k1_sensors.dim() == 4:
                k1_sensors = k1_sensors.squeeze(1)         # -> [B, N_s, D]

            # Midpoint
            cls_mid = cls_cur + dt_eff * k1_cls              # [B, 1, D]
            sensors_mid = sensors_cur + dt_eff * k1_sensors  # [B, N_s, D]

            # Evaluate at midpoint (no append to avoid double-growth)
            cls_mid_y, sensors_mid_y = self._step_layers(
                self._add_pos(cls_mid.unsqueeze(1), pos_idx).squeeze(1),
                self._add_pos(sensors_mid.unsqueeze(1), pos_idx).squeeze(1),
                imp,
                append=False  # New: No-append for midpoint
            )

            k2_cls = self.cls_head(cls_mid_y)              # [B, 1, D]
            k2_sensors = self.sensor_head(sensors_mid_y)   # [B, 1, N_s, D]
            if k2_sensors.dim() == 4:
                k2_sensors = k2_sensors.squeeze(1)         # -> [B, N_s, D]

            # Heun update
            cls_next = cls_cur + 0.5 * dt_eff * (k1_cls + k2_cls)                   # [B, 1, D]
            sensors_next = sensors_cur + 0.5 * dt_eff * (k1_sensors + k2_sensors)   # [B, N_s, D]
            latent_next = torch.cat([cls_next, sensors_next], dim=1)                # [B, 1 + N_s, D]

            outputs.append(latent_next)

            # Teacher forcing
            use_tf = (teacher_force_seq is not None and step < teacher_force_seq.size(1) and
                      torch.rand((), device=dev) < teacher_force_prob)
            latent_cur = teacher_force_seq[:, step] if use_tf else latent_next
            cls_cur = latent_cur[:, :1, :]
            sensors_cur = latent_cur[:, 1:, :]

            # Truncated BPTT
            if truncate_k and ((step + 1) % truncate_k == 0):
                for l in range(self.n_layers):
                    for attn in [self.cls_self_attns[l], 
                                 self.cls_cross_attns[l],
                                #  self.sensor_self_attns[l], 
                                 self.sensor_cross_attns[l]
                                 ]:
                        if attn.K is not None:
                            attn.K = attn.K.detach()
                            attn.V = attn.V.detach()

        return torch.stack(outputs, 1)  # [B, N_fore, 1 + N_s, D]

    # ------------------------------------------------------------------
    # Greedy generation (no grad) – evaluation / inference
    # No-grad autoregressive, hierarchical
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, obs_window: torch.Tensor, N_fore: int, imp: torch.Tensor | None = None) -> torch.Tensor:
        """
        obs_window: [B, T_obs, 1 + N_s, D]
        imp: [B, N_s]
        returns: [B, N_fore, 1 + N_s, D]
        """
        B, T_obs, M, _ = obs_window.shape
        N_s = M - 1
        dev = obs_window.device
        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=obs_window.device, dtype=obs_window.dtype)

        # Reset states
        for l in range(self.n_layers):
            self.cls_self_attns[l].reset_state(B, dev)
            self.cls_cross_attns[l].reset_state(B, dev)
            # self.sensor_self_attns[l].reset_state(B * N_s, dev)
            self.sensor_cross_attns[l].reset_state(B, dev)

        out = []
        for t in range(T_obs):
            token_raw = obs_window[:, t]
            token_pe = self._add_pos(token_raw.unsqueeze(1), t).squeeze(1)
            cls_t = token_pe[:, :1, :]
            sensors_t = token_pe[:, 1:, :]
            _ = self._step_layers(cls_t, sensors_t, imp)
            out.append(token_raw)

        dt = self._effective_dt()
        latent_cur = obs_window[:, -1]
        cls_cur = latent_cur[:, :1, :]
        sensors_cur = latent_cur[:, 1:, :]

        for step in range(N_fore - T_obs):
            pos_idx = T_obs + step
            cls_y, sensors_y = self._step_layers(
                self._add_pos(cls_cur.unsqueeze(1), pos_idx).squeeze(1),
                self._add_pos(sensors_cur.unsqueeze(1), pos_idx).squeeze(1),
                imp,
                append=True  # Append for k1
            )

            # k1
            k1_cls = self.cls_head(cls_y)                  # [B, 1, D]
            k1_sensors = self.sensor_head(sensors_y)       # [B, 1, N_s, D]
            if k1_sensors.dim() == 4:
                k1_sensors = k1_sensors.squeeze(1)         # -> [B, N_s, D]
            # Midpoint
            cls_mid = cls_cur + dt * k1_cls
            sensors_mid = sensors_cur + dt * k1_sensors
            # Evaluate at midpoint (no append)
            cls_mid_y, sensors_mid_y = self._step_layers(
                self._add_pos(cls_mid.unsqueeze(1), pos_idx).squeeze(1),
                self._add_pos(sensors_mid.unsqueeze(1), pos_idx).squeeze(1),
                imp,
                append=False  # New: No-append for midpoint
            )
            # k2
            k2_cls = self.cls_head(cls_mid_y)              # [B, 1, D]
            k2_sensors = self.sensor_head(sensors_mid_y)   # [B, 1, N_s, D]
            if k2_sensors.dim() == 4:
                k2_sensors = k2_sensors.squeeze(1)         # -> [B, N_s, D]
            # Update
            cls_cur = cls_cur + 0.5 * dt * (k1_cls + k2_cls)
            sensors_cur = sensors_cur + 0.5 * dt * (k1_sensors + k2_sensors)

            latent_cur = torch.cat([cls_cur, sensors_cur], dim=1)  # [B, 1 + N_s, D]
            out.append(latent_cur)

        return torch.stack(out, 1)

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    def _forward_no_ckpt(self, cls: torch.Tensor, sensors: torch.Tensor, imp: torch.Tensor) -> tuple:
        for idx in range(self.n_layers):
            cls, sensors = self._block(idx, cls, sensors, imp, incremental=False)
        return cls, sensors

    def _step_layers(self, cls_t: torch.Tensor, sensors_t: torch.Tensor, imp: torch.Tensor, append: bool = True) -> tuple:
        for idx in range(self.n_layers):
            cls_t, sensors_t = self._block(idx, cls_t, sensors_t, imp, incremental=True, append=append)  # New: Pass append
        return cls_t, sensors_t

# Revised 0922
class TemporalDecoderHierarchical(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 4096,
        dt: float = 0.02,
        learnable_dt: bool = False,
        dropout: float = 0.0,
        rope_base: float = 1000.0,
        checkpoint_every_layer: bool = True,
        imp_threshold: float = 0.1,  # Mask if imp < threshold

        n_window: int = 16,       # Fixed window size for history
        pooling_kernel: int = 2,  # Kernel and stride for pooling
        pooling_layers: int = 2,  # Number of pooling applications to compress (e.g., 64 -> 32 -> 16)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.imp_threshold = imp_threshold
        self.n_window = n_window
        self.pooling_kernel = pooling_kernel
        self.pooling_layers = pooling_layers

        # Hierarchical layers (Stack n_layers of hierarchical blocks)
        self.cls_self_attns     = nn.ModuleList([MultiheadSoftmaxAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.cls_cross_attns    = nn.ModuleList([CAU_MultiheadCrossAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        # self.sensor_self_attns  = nn.ModuleList([MultiheadSoftmaxAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.sensor_cross_attns = nn.ModuleList([CAU_MultiheadCrossAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

        self.pre_norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])  # For CLS and sensors
        self.post_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
            ) for _ in range(n_layers * 2)  # For CLS and sensors
        ])

        # Heun heads (separate for CLS and sensors)
        self.cls_head    = nn.Linear(d_model, d_model)
        self.sensor_head = nn.Linear(d_model, d_model)

        # Learnable positional embedding (time)
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # Learnable spatial embedding for sensors (applied per-sensor)
        self.spatial_emb = nn.Linear(d_model, d_model)  # Projects sensor features with spatial awareness

        # Pooling layers (separate for CLS and sensors; avg and max, then combine)
        self.cls_avg_pool = nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.cls_max_pool = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.sensor_avg_pool = nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.sensor_max_pool = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)

        # Post-transformer refinement (1D conv + FC, separate for CLS and sensors)
        conv_dim = d_model  # Assuming after pooling, time dim is small
        self.cls_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.cls_fc = nn.Linear(d_model, d_model)
        self.sensor_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.sensor_fc = nn.Linear(d_model, d_model)

        # dt handling 
        self.register_buffer("dt_const", torch.tensor(dt))
        if learnable_dt:
            self.dt_scale = nn.Parameter(torch.zeros(()))
        else:
            self.dt_scale = None

        self.use_ckpt = checkpoint_every_layer

    def apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
        x:  [B, T, M, D]  or [B, M, D] or [B, D]
        pos: (1, T, 1) integer/float time indices
        """
        assert x.dim() in (2, 3, 4), f"apply_rope: unsupported x.dim={x.dim()}"
        orig_dim = x.dim()

        # Lift to 4D: [B, T, M, D]
        if orig_dim == 2:          # [B, D] -> [B, 1, 1, D]
            x = x.unsqueeze(1).unsqueeze(1)
        elif orig_dim == 3:        # [B, M, D] -> [B, 1, M, D]
            x = x.unsqueeze(1)
        # else: already [B, T, M, D]

        B, T, M, D = x.shape
        assert D % 2 == 0, f"apply_rope: D must be even, got D={D}"
        d = D // 2

        # Frequencies and angles
        freq = self.rope_base ** (-torch.arange(d, device=x.device, dtype=x.dtype) / d)  # [d]
        # pos: [1, T, 1] -> [1, T, 1, 1], broadcast with freq -> [1, T, 1, d]
        angle = pos.to(x.dtype).unsqueeze(-1) * freq.view(1, 1, 1, d)

        s, c = angle.sin(), angle.cos()
        x1, x2 = x[..., :d], x[..., d:]  # [B, T, M, d]
        xr = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)  # [B, T, M, D]

        # Restore original rank
        if orig_dim == 2: xr = xr.squeeze(1).squeeze(1)    # -> [B, D]
        elif orig_dim == 3: xr = xr.squeeze(1)             # -> [B, M, D]

        return xr

    def _add_pos(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Adds rotary position embeddings to x. Accepts x with shape:
        - [B, T, M, D]  (T steps)
        - [B, M, D]     (T=1)
        - [B, D]        (T=1, M=1)
        """
        if x.dim() == 4:
            T = x.shape[1]
        elif x.dim() in (2, 3):
            T = 1
        else:
            raise ValueError(f"_add_pos: unsupported x.dim={x.dim()}")
        pos = torch.arange(offset, offset + T, device=x.device, dtype=x.dtype).view(1, T, 1)  # [1, T, 1]
        return self.apply_rope(x, pos)

    def _ensure_imp(self, imp, *, B: int, N_s: int, device, dtype=torch.float32):
        if imp is None:
            return torch.ones(B, N_s, device=device, dtype=dtype)
        return imp

    def _effective_dt(self):
        if self.dt_scale is None:
            return self.dt_const
        return self.dt_const * F.softplus(self.dt_scale)

    # ------------------------------------------------------------------
    # Process window: Embeddings + Separate Pooling for CLS/Sensors
    # ------------------------------------------------------------------
    def _process_window(self, window: torch.Tensor, offset: int, imp: torch.Tensor) -> tuple:
        """
        window: [B, T_win, 1 + N_s, D]
        Returns pooled cls and sensors: [B, T_pooled, 1, D] and [B, T_pooled, N_s, D]
        """
        B, T_win, M, D = window.shape
        N_s = M - 1

        # Add time embeddings (RoPE)
        window = self._add_pos(window, offset)

        # Separate CLS and sensors
        cls_win = window[:, :, :1, :]      # [B, T, 1, D]
        sensors_win = window[:, :, 1:, :]  # [B, T, N_s, D]

        # Spatial embedding once (per time, per sensor)
        sensors_win = self.spatial_emb(sensors_win)

        # Apply pooling layers separately (multiple times to compress)
        for _ in range(self.pooling_layers):
            # Current time length (same for cls and sensors)
            T_cur = cls_win.shape[1]
            # If window is shorter than kernel, stop pooling
            if T_cur < self.pooling_kernel:
                break

            # ---- CLS path ----
            # [B, T_cur, 1, D] -> [B, D, T_cur]
            cls_t = cls_win.squeeze(2).permute(0, 2, 1)        # [B, D, T_cur]
            cls_avg = self.cls_avg_pool(cls_t)                 # [B, D, T_next]
            cls_max = self.cls_max_pool(cls_t)                 # [B, D, T_next]
            cls_pooled = (cls_avg + cls_max) / 2
            # Back to [B, T_next, 1, D]
            cls_win = cls_pooled.permute(0, 2, 1).unsqueeze(2) # [B, T_next, 1, D]

            # ---- Sensors path ----
            # sensors_win: [B, T_cur, N_s, D]
            B_, T_cur_s, N_s_, D_ = sensors_win.shape
            assert B_ == B and T_cur_s == T_cur and N_s_ == N_s and D_ == D, f"Unexpected sensors_win shape {sensors_win.shape}"

            # Pool along time per sensor: [B*N_s, D, T_cur]
            sensors_t = sensors_win.permute(0, 2, 3, 1).reshape(B * N_s, D, T_cur)
            sensors_avg = self.sensor_avg_pool(sensors_t)      # [B*N_s, D, T_next]
            sensors_max = self.sensor_max_pool(sensors_t)      # [B*N_s, D, T_next]
            sensors_pooled = (sensors_avg + sensors_max) / 2   # [B*N_s, D, T_next]

            # Back to [B, T_next, N_s, D]
            T_next = sensors_pooled.shape[-1]
            sensors_win = sensors_pooled.reshape(B, N_s, D, T_next).permute(0, 3, 1, 2)  # [B, T_next, N_s, D]

        return cls_win, sensors_win

    # ------------------------------------------------------------------
    # Hierarchical Block (modified for parallel window processing, no incremental)
    # ------------------------------------------------------------------
    def _block(
        self,
        idx: int,
        cls: torch.Tensor,
        sensors: torch.Tensor,
        imp: torch.Tensor,
    ) -> tuple:
        """
        Modified for parallel: cls [B, T_pooled, 1, D], sensors [B, T_pooled, N_s, D]
        Uses forward() of attentions instead of step().
        """

        # -----------------------------------------------------------
        # 1.  Resolve shapes
        # -----------------------------------------------------------
        B, T, _, D = cls.shape
        _, _, N_s, _ = sensors.shape

        # -----------------------------------------------------------
        # 2.  CLS stream
        # -----------------------------------------------------------
        cls_residual = cls
        cls_normed = self.pre_norms[idx](cls)  # [B, T, 1, D]

        cls_squeezed = cls_normed.squeeze(2)  # [B, T, D]
        cls_attn = self.cls_self_attns[idx](cls_squeezed).unsqueeze(2)  # [B, T, 1, D]

        cls_cross = self.cls_cross_attns[idx](cls_normed, sensors, imp_weights = imp)   # CLS attend to Sensors need imp
        cls = cls_attn + cls_cross
        
        cls = self.post_norms[idx](cls_residual + cls)
        cls = cls + self.ffns[idx](cls)

        # -----------------------------------------------------------
        # 3.  Sensor stream
        # -----------------------------------------------------------
        sensors = self.pre_norms[idx + self.n_layers](sensors)  # [B, T, N_s, D]

        sensors_sa = sensors.view(B * N_s, T, D)  # [B*N_s, T, D]
        # sensors_sa = self.sensor_self_attns[idx](sensors_sa)  # Uncomment if needed
        sensors = sensors_sa.view(B, T, N_s, D)

        sensors = sensors + self.sensor_cross_attns[idx](sensors, cls, imp_weights = None) # Sensor attend to CLS do not need imp

        sensors = self.post_norms[idx + self.n_layers](sensors)
        sensors = sensors + self.ffns[idx + self.n_layers](sensors)

        # -----------------------------------------------------------
        # 4.  Importance masking
        # -----------------------------------------------------------
        # mask = (imp.unsqueeze(1).unsqueeze(-1) >= self.imp_threshold)  # [B, 1, N_s, 1]
        # sensors = sensors * mask.float().unsqueeze(1)  # Broadcast over T

        return cls, sensors

    def _block_ckpt(self, idx: int, cls: torch.Tensor, sensors: torch.Tensor, imp: torch.Tensor) -> tuple:
        if not (self.use_ckpt and self.training):
            return self._block(idx, cls, sensors, imp)

        def fn(c, s, i):
            return self._block(idx, c, s, i)
        return cp.checkpoint(fn, cls, sensors, imp, use_reentrant=False)

    # ------------------------------------------------------------------
    # Post-transformer refinement (conv + fc, separate for CLS/sensors)
    # ------------------------------------------------------------------
    def _refine(self, cls: torch.Tensor, sensors: torch.Tensor) -> tuple:
        """
        cls:     [B, T, 1, D] -> [B, T, 1, D]
        sensors: [B, T, N_s, D] -> [B, T, N_s, D]
        """
        B, T, _, D = cls.shape
        _, T2, N_s, D2 = sensors.shape
        assert T2 == T and D2 == D, f"Shape mismatch: cls T={T}, sensors T={T2}; D={D}, sensors D={D2}"

        # ---- CLS path ----
        cls_t = cls.squeeze(2).permute(0, 2, 1)          # [B, D, T]
        cls_ref = F.relu(self.cls_conv(cls_t))           # [B, D, T]
        cls_ref = cls_ref.permute(0, 2, 1).unsqueeze(2)  # [B, T, 1, D]
        cls_ref = self.cls_fc(cls_ref)                   # [B, T, 1, D]

        # ---- Sensors path ----
        # [B, T, N_s, D] -> [B*N_s, D, T]
        sensors_t = sensors.permute(0, 2, 3, 1).reshape(B * N_s, D, T)
        sensors_ref = F.relu(self.sensor_conv(sensors_t))    # [B*N_s, D, T]

        # Back to [B, T, N_s, D]
        sensors_ref = sensors_ref.permute(0, 2, 1)           # [B*N_s, T, D]
        sensors_ref = sensors_ref.reshape(B, N_s, T, D)      # [B, N_s, T, D]
        sensors_ref = sensors_ref.permute(0, 2, 1, 3)        # [B, T, N_s, D]
        sensors_ref = self.sensor_fc(sensors_ref)            # [B, T, N_s, D]

        return cls_ref, sensors_ref

    def forward(self, x_seq: torch.Tensor, imp: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_seq: [B, T, 1 + N_s, D]
        imp: [B, N_s]
        """
        B, T, M, D = x_seq.shape
        N_s = M - 1

        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=x_seq.device, dtype=x_seq.dtype)

        # Treat full x_seq as "window"
        cls_pooled, sensors_pooled = self._process_window(x_seq, 0, imp)

        for idx in range(self.n_layers):
            cls_pooled, sensors_pooled = self._block_ckpt(idx, cls_pooled, sensors_pooled, imp)

        # Refine
        cls_pooled, sensors_pooled = self._refine(cls_pooled, sensors_pooled)

        # Hierarchical Heun
        dt = self._effective_dt()
        k1_cls = self.cls_head(cls_pooled)
        k1_sensors = self.sensor_head(sensors_pooled)
        cls_mid = cls_pooled + dt * k1_cls
        sensors_mid = sensors_pooled + dt * k1_sensors
        # Pass mid through no-ckpt forward (re-process mid as mini-window of size 1)
        cls_mid, sensors_mid = self._forward_no_ckpt(cls_mid, sensors_mid, imp)
        cls_mid, sensors_mid = self._refine(cls_mid, sensors_mid)
        k2_cls = self.cls_head(cls_mid)
        k2_sensors = self.sensor_head(sensors_mid)
        cls_next = cls_pooled + 0.5 * dt * (k1_cls + k2_cls)
        sensors_next = sensors_pooled + 0.5 * dt * (k1_sensors + k2_sensors)

        return torch.cat([cls_next, sensors_next], dim=2)  # [B, T_pooled, 1 + N_s, D] (note: output T may be compressed)

    def rollout_with_grad(
        self,
        obs_window: torch.Tensor,  # [B, T_obs, 1 + N_s, D]
        N_fore: int,
        imp: torch.Tensor | None = None,  # [B, N_s]
        *,
        truncate_k: int | None = 64,
        teacher_force_seq: torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ) -> torch.Tensor:
        assert self.training, "Call only in training mode"
        B, T_obs, M, D = obs_window.shape
        N_s = M - 1
        dev = obs_window.device
        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=obs_window.device, dtype=obs_window.dtype)

        # Initialize rolling window as deque (maxlen=n_window)
        history = deque(maxlen=self.n_window)
        for t in range(T_obs):
            history.append(obs_window[:, t])  # [B, 1 + N_s, D]

        outputs = [torch.stack(list(history), dim=1)] if len(history) > 0 else []  # Initial outputs from obs

        dt_eff = self._effective_dt()
        steps_left = N_fore - T_obs
        latent_cur = history[-1] if history else torch.zeros(B, 1 + N_s, D, device=dev)  # Fallback
        cls_cur = latent_cur[:, :1, :]
        sensors_cur = latent_cur[:, 1:, :]

        for step in range(steps_left):
            # Extract current window [B, min(n_window, len(history)), 1 + N_s, D]
            win_tensor = torch.stack(list(history), dim=1) if len(history) > 0 else latent_cur.unsqueeze(1)
            pos_offset = T_obs + step - win_tensor.shape[1] + 1  # Align positions

            # Process window
            cls_win, sensors_win = self._process_window(win_tensor, pos_offset, imp)

            # Hierarchical blocks on pooled window
            for idx in range(self.n_layers):
                cls_win, sensors_win = self._block_ckpt(idx, cls_win, sensors_win, imp)

            # Refine
            cls_y, sensors_y = self._refine(cls_win, sensors_win)

            # Heun k1 (use last time step of refined as y)
            k1_cls = self.cls_head(cls_y[:, -1:, :, :]).squeeze(1) # [B, 1, D]
            k1_sensors = self.sensor_head(sensors_y[:, -1:, :, :]).squeeze(1) # [B, N_s, D]

            # Midpoint (treat as single-step window)
            cls_mid_token = cls_cur + dt_eff * k1_cls # [B, 1, D]
            sensors_mid_token = sensors_cur + dt_eff * k1_sensors # [B, N_s, D]
            mid_window = torch.cat([cls_mid_token, sensors_mid_token], dim=1).unsqueeze(1) # [B, 1, 1+N_s, D]

            cls_mid, sensors_mid = self._process_window(mid_window, pos_offset + win_tensor.shape[1] - 1, imp)
            for idx in range(self.n_layers):
                cls_mid, sensors_mid = self._block_ckpt(idx, cls_mid, sensors_mid, imp)
            cls_mid_y, sensors_mid_y = self._refine(cls_mid, sensors_mid)

            k2_cls = self.cls_head(cls_mid_y[:, -1:, :, :]).squeeze(1) # [B, 1, D]
            k2_sensors = self.sensor_head(sensors_mid_y[:, -1:, :, :]).squeeze(1) # [B, N_s, D]

            # Heun update
            cls_next = cls_cur + 0.5 * dt_eff * (k1_cls + k2_cls) # [B, 1, D]
            sensors_next = sensors_cur + 0.5 * dt_eff * (k1_sensors + k2_sensors) # [B, N_s, D]
            latent_next = torch.cat([cls_next, sensors_next], dim=1) # [B, 1 + N_s, D]

            # Append to outputs and history
            # outputs.append(latent_next)
            outputs.append(latent_next.unsqueeze(1)) # [B, 1, 1+N_s, D]
            history.append(latent_next)

            # Teacher forcing
            use_tf = (teacher_force_seq is not None and step < teacher_force_seq.size(1) and
                      torch.rand((), device=dev) < teacher_force_prob)
            if use_tf:
                history[-1] = teacher_force_seq[:, step]  # Override last in history

            # Truncated BPTT (simulate by detaching history every truncate_k steps)
            if truncate_k and ((step + 1) % truncate_k == 0):
                history = deque([h.detach() for h in history], maxlen=self.n_window)

            # Update current
            latent_cur = history[-1]
            cls_cur = latent_cur[:, :1, :]
            sensors_cur = latent_cur[:, 1:, :]

        # return torch.stack(outputs, 1)  # [B, N_fore, 1 + N_s, D]
        return torch.cat(outputs, dim=1) # [B, T_obs + steps_left, 1+N_s, D]

    # ------------------------------------------------------------------
    # Greedy generation (no grad) – evaluation / inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, obs_window: torch.Tensor, N_fore: int, imp: torch.Tensor | None = None) -> torch.Tensor:
        """
        obs_window: [B, T_obs, 1 + N_s, D]
        imp: [B, N_s]
        returns: [B, T_obs + steps_left, 1 + N_s, D]  (same as training; slice if you only want forecasts)
        """
        B, T_obs, M, D = obs_window.shape
        N_s = M - 1
        dev = obs_window.device
        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=obs_window.device, dtype=obs_window.dtype)

        # Rolling window seeded with observed tokens
        history = deque(maxlen=self.n_window)
        for t in range(T_obs):
            history.append(obs_window[:, t])  # [B, 1 + N_s, D]

        # Collect outputs as time-major blocks; start with observed segment
        outputs = [torch.stack(list(history), dim=1)] if len(history) > 0 else []

        dt = self._effective_dt()
        steps_left = N_fore - T_obs

        latent_cur = history[-1] if history else torch.zeros(B, 1 + N_s, D, device=dev, dtype=obs_window.dtype)
        cls_cur = latent_cur[:, :1, :]      # [B, 1, D]
        sensors_cur = latent_cur[:, 1:, :]  # [B, N_s, D]

        for step in range(steps_left):
            # Current window [B, T_win, 1 + N_s, D]
            win_tensor = torch.stack(list(history), dim=1) if len(history) > 0 else latent_cur.unsqueeze(1)
            pos_offset = T_obs + step - win_tensor.shape[1] + 1

            # Context processing
            cls_win, sensors_win = self._process_window(win_tensor, pos_offset, imp)
            for idx in range(self.n_layers):
                cls_win, sensors_win = self._block(idx, cls_win, sensors_win, imp)  # no checkpoint in no_grad
            cls_y, sensors_y = self._refine(cls_win, sensors_win)

            # Heun k1 from last pooled timestep (reduce to single step for the ODE state)
            k1_cls = self.cls_head(cls_y[:, -1:, :, :]).squeeze(1)           # [B, 1, D]
            k1_sensors = self.sensor_head(sensors_y[:, -1:, :, :]).squeeze(1) # [B, N_s, D]

            # Midpoint: build a one-step window explicitly [B, 1, 1+N_s, D]
            cls_mid_token = cls_cur + dt * k1_cls                 # [B, 1, D]
            sensors_mid_token = sensors_cur + dt * k1_sensors     # [B, N_s, D]
            mid_window = torch.cat([cls_mid_token, sensors_mid_token], dim=1).unsqueeze(1)  # [B, 1, 1+N_s, D]

            cls_mid, sensors_mid = self._process_window(mid_window, pos_offset + win_tensor.shape[1] - 1, imp)
            for idx in range(self.n_layers):
                cls_mid, sensors_mid = self._block(idx, cls_mid, sensors_mid, imp)
            cls_mid_y, sensors_mid_y = self._refine(cls_mid, sensors_mid)

            # Heun k2 (also single-step)
            k2_cls = self.cls_head(cls_mid_y[:, -1:, :, :]).squeeze(1)           # [B, 1, D]
            k2_sensors = self.sensor_head(sensors_mid_y[:, -1:, :, :]).squeeze(1) # [B, N_s, D]

            # Heun update on single-step state
            cls_next = cls_cur + 0.5 * dt * (k1_cls + k2_cls)              # [B, 1, D]
            sensors_next = sensors_cur + 0.5 * dt * (k1_sensors + k2_sensors)  # [B, N_s, D]
            latent_next = torch.cat([cls_next, sensors_next], dim=1)       # [B, 1 + N_s, D]

            # Append to outputs and update buffers
            outputs.append(latent_next.unsqueeze(1))  # [B, 1, 1+N_s, D]
            history.append(latent_next)

            # Advance current state
            cls_cur = cls_next
            sensors_cur = sensors_next

        # Concatenate time blocks: [B, T_obs + steps_left, 1 + N_s, D]
        return torch.cat(outputs, dim=1)

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    def _forward_no_ckpt(self, cls: torch.Tensor, sensors: torch.Tensor, imp: torch.Tensor) -> tuple:
        for idx in range(self.n_layers):
            cls, sensors = self._block(idx, cls, sensors, imp)
        return cls, sensors

# ================================================================
# Temporal Decoder Adapter for all modules
# ================================================================
class TemporalDecoderAdapter(nn.Module):
    def __init__(self, core: TemporalDecoderLinear):
        super().__init__()
        self.core = core

    def forward_autoreg(
        self,
        G_latent          : torch.Tensor,      # (B, T_obs, D)
        N_Fore            : int,
        N_window          : int,
        *,
        imp: torch.Tensor | None = None,       # [B, N_s]
        truncate_k        : int | None = 32,
        teacher_force_seq : torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ):
        """
        Pure autoregressive rollout that *keeps* gradients.
        Call this only with model.train() set.
        """
        assert self.training, "Use only in training mode"
        if imp is not None:
            output = self.core.rollout_with_grad(
                obs_window        = G_latent[:, :N_window],
                N_fore            = N_Fore,
                imp               = imp,
                truncate_k        = truncate_k,
                teacher_force_seq = teacher_force_seq,
                teacher_force_prob= teacher_force_prob,
            )
        else:
            output = self.core.rollout_with_grad(
                obs_window        = G_latent[:, :N_window],
                N_fore            = N_Fore,
                truncate_k        = truncate_k,
                teacher_force_seq = teacher_force_seq,
                teacher_force_prob= teacher_force_prob,
            )            
        # Handle if output is (traj, traj_logvar) or just traj
        if isinstance(output, tuple) and len(output) == 2:
            return output  # (traj, traj_logvar)
        elif isinstance(output, torch.Tensor):
            return output, None  # (traj, None)
        else:
            raise ValueError(f"Unexpected output from core.rollout_with_grad: {type(output)}")

    def forward(
            self, G_latent: torch.Tensor, 
            N_Fore: int, 
            N_window: int, 
            imp: torch.Tensor | None = None,  # [B, N_s]
            ):
        
        if imp is not None:
            output = self.core.generate(G_latent[:, :N_window], N_Fore, imp)
        else:
            output = self.core.generate(G_latent[:, :N_window], N_Fore)
        
        # Handle if output is (traj, traj_logvar) or just traj
        if isinstance(output, tuple) and len(output) == 2:
            return output  # (traj, traj_logvar)
        elif isinstance(output, torch.Tensor):
            return output, None  # (traj, None)
        else:
            raise ValueError(f"Unexpected output from core.generate: {type(output)}")

# ---------------------------------------------------------------------
# Perceiver-based reconstructor without domain soft domain decomposition
# ---------------------------------------------------------------------
class FlashCrossAttention(nn.Module):
    """
    Memory-efficient multi-head cross-attention.

    Uses torch 2.x scaled_dot_product_attention (Flash / Helium / Math).
    Optional query-chunking to keep peak memory below
        B · block_size · d_model   instead of B · P · d_model.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 block_size: int | None = None):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.block_size  = block_size
        self.dropout_p   = dropout

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out  = nn.Linear(dim, dim, bias=False)

    # -------------------------------------------------------------
    def _flash_attn(self, q, k, v):
        """
        Wrapper around torch.scaled_dot_product_attention that:
        - uses dropout only during training
        - works with fp16 / bf16
        """
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False)

    # -------------------------------------------------------------
    def forward(self, q_in, k_in, v_in):
        """
        q_in : [B, P, D]       (queries = coordinates)
        k/v  : [B, L, D]       (latent tokens)
        returns same shape as q_in
        """
        B, P, _ = q_in.shape
        L       = k_in.size(1)

        # 1. project once ----------------------------------------------------
        q = self.to_q(q_in)   # [B,P,D]
        k = self.to_k(k_in)   # [B,L,D]
        v = self.to_v(v_in)   # [B,L,D]

        # 2. reshape to [B,h,*,d_h] without extra copies ---------------------
        def split(x):                                         
            B, N, _ = x.shape                                 
            return x.view(B, N, self.num_heads, self.head_dim) \
                    .transpose(1, 2)                            # [B,h,N,d_h]

        q = split(q); k = split(k); v = split(v)

        # 3a. fast path – no chunking ---------------------------------------
        if self.block_size is None or P <= self.block_size:
            out = self._flash_attn(q, k, v)                    # [B,h,P,d_h]

        # 3b. slow path – stream the queries -------------------------------
        else:
            chunks = []
            for p0 in range(0, P, self.block_size):
                p1  = min(P, p0 + self.block_size)
                q_b = q[:, :, p0:p1]                           # [B,h,b,d_h]
                o_b = self._flash_attn(q_b, k, v)              # [B,h,b,d_h]
                chunks.append(o_b)
            out = torch.cat(chunks, dim=2)                     # [B,h,P,d_h]

        # 4. merge heads -----------------------------------------------------
        out = out.transpose(1, 2).contiguous()                 # [B,P,h,d_h]
        out = out.view(B, P, self.dim)                         # [B,P,D]
        return self.out(out)                                   # final proj

class PerceiverReconstructor(nn.Module):
    """
    Given               z : (B, T, D) latent trajectory
                        Y : (B, P, 2/3) query coordinates
    return        u_hat : (B, T, P, N_channels)
    Memory/FLOPs grow O(P) thanks to latent cross-attention.
    """
    def __init__(self,
                 d_model      : int,
                 num_heads      : int,
                 N_channels     : int,
                 pe_module      : FourierEmbedding | None = None,

                 num_freqs      : int = 64,
                 dropout        : float = 0.0,
                 share_pe       : bool  = False):
        """
        pe_module :   pass the *same* FourierEmbedding instance from the
                      encoder to share weights; if None, a fresh one is built.
        """
        super().__init__()
        # ------------------- positional encoder -------------------------
        if pe_module is None:
            self.pe = FourierEmbedding(in_dim=2,
                                       num_frequencies=num_freqs,
                                       learnable=True,
                                       sigma=10.0)
        else:
            self.pe = pe_module if share_pe else pe_module.__class__(
                         in_dim          = 2,
                         num_frequencies = pe_module.B.shape[1],
                         learnable       = True,
                         sigma           = 1.0)
            if not share_pe:
                # start from same weights but allow to diverge
                self.pe.load_state_dict(pe_module.state_dict())

        self.coord_proj = nn.Linear(self.pe.B.shape[1]*2, d_model)
        self.lat_proj   = nn.Linear(d_model, d_model)

        # ------------------ cross-attention --------------------
        self.cross_attn  = CrossAttention(d_model, num_heads, dropout)
        self.norm        = nn.LayerNorm(d_model)

        # 1-hidden-layer MLP
        self.mlp = GEGLU(d_model, mult=4)
        self.head = nn.Linear(d_model, N_channels)

        self.dropout = nn.Dropout(0.1) 
        self.d_model = d_model
        self.N_channels = N_channels

    # ===================================================================
    def forward(self, z: torch.Tensor, Y: torch.Tensor):
        """
        z : (B, T, D) or (B, T, L, D)
        Y : (B, P, 2/3)
        """
        B = z.shape[0]
        T = z.shape[1]
        P = Y.size(1)

        # coordinate tokens – broadcast to all T
        coord_tok = self.coord_proj(self.pe(Y))              # (B,P,Dm)

        # pre-allocate output tensor (avoids Python list) ------------
        out_all = torch.empty(B, T, P, self.N_channels,
                              device=z.device,
                              dtype=z.dtype)

        # loop over time slices (keeps only one slice of activations)
        for t in range(T):
            slice_t = z[:, t, ...]                       # [B,D] or [B,L,D]
            if slice_t.dim() == 2:
                slice_t = slice_t.unsqueeze(1)           # [B,1,D]

            lat_tok = self.lat_proj(slice_t)             # [B,L,D]

            x = self.cross_attn(coord_tok, lat_tok, lat_tok)  # [B,P,D]
            x = self.norm(x + coord_tok)                      # residual-1
            x = x + self.mlp(x)                               # residual-2
            out_all[:, t] = self.head(x)

        return out_all                                        # [B,T,P,C]

# ---------------------------------------------------------------------
# Domain Adaptive Reconstructor with soft boundaries
# ---------------------------------------------------------------------

#   Revised 0921: Weighted Fusion in Aggregation
class SoftDomainAdaptiveReconstructor(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        N_channels: int,

        num_freqs: int = 64,
        dropout: float = 0.0,

        overlap_ratio: float = 0.05,
        importance_scale: float = 0.50,

        bandwidth_init: float = 0.05,
        top_k: int | None = None,
        per_sensor_sigma: bool = False,
        CalRecVar: bool = False, 
        retain_cls: bool = False,
        use_checkpoint: bool = True,

        # --- New for phi incorporation toggles (set to True for combined use) ---
        use_weighted_fusion: bool = True,   # Toggle Weighted Fusion in Aggregation
        phi_scale: float = 0.5,             # Tunable scale for phi modulation (to avoid over-amplification)
    ):
        super().__init__()

        # ------------------- positional encoder -------------------------
        self.pe = FourierEmbedding(in_dim=2,  
                                   num_frequencies=num_freqs,
                                   learnable=True,
                                   sigma=10.0)

        self.coord_proj = nn.Linear(self.pe.B.shape[1] * 2, d_model)
        self.lat_proj = nn.Linear(d_model, d_model)

        # ------------------ cross-attention --------------------
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

        # 1-hidden-layer MLP 
        self.mlp = GEGLU(d_model, mult=4)
        self.head = nn.Linear(d_model, N_channels)

        self.dropout = nn.Dropout(0.1)
        self.d_model = d_model
        self.N_channels = N_channels

        self.overlap_ratio    = overlap_ratio
        self.importance_scale = importance_scale
        self.CalRecVar = CalRecVar

        self.block_size        = 256         
        self.top_k             = top_k
        self.per_sensor_sigma  = per_sensor_sigma
        self.register_parameter("log_sigma", None)  
        self.bandwidth_init    = bandwidth_init
        self._prev_S           = None

        self.coord_norm = nn.RMSNorm(d_model)
        self.agg_norm   = nn.RMSNorm(d_model)
        self.mlp_norm   = nn.RMSNorm(d_model)

        # flag for retaining CLS and hierarchical reconstruction
        self.retain_cls = retain_cls
        print(f'self.retain_cls is {self.retain_cls}')
        if self.retain_cls: 
            self.fusion_proj = nn.Linear(2 * d_model, d_model)  # Projects concatenated [local + CLS] back to d_model

        if CalRecVar:
            self.stats_dim = 4  # mean_d, std_d, effective_K, mean_phi
            self.stats_proj = nn.Linear(self.stats_dim, 16)  # Project to embed dim; concatenate to x
            self.var_head = nn.Linear(self.d_model + 16, self.N_channels)  # Input now larger

        # --- params for phi incorporation ---
        self.use_weighted_fusion = use_weighted_fusion
        self.phi_scale = phi_scale  # Scale factor to control phi's influence
        self.use_checkpoint = use_checkpoint

    @staticmethod
    @torch.jit.ignore
    def _topk_aggregate(lat_proj:  torch.Tensor,   # (B,T,S,D)
                        top_idx:   torch.Tensor,   # (B,P,K)
                        weights_k: torch.Tensor,   # (B,P,K)

                        CalRecVar : bool,
                        d_k: torch.Tensor,         # Pass d_k (B,P,K)
                        phi_k: torch.Tensor,       # Pass gathered phi (B,P,K); phi expanded per query
                        valid_k: torch.Tensor,     # Pass valid (B,P,K)

                        S: int) -> torch.Tensor:
        """
        Return h[b,t,p] = Σ_k w[b,p,k] · φ[b,t, top_idx[b,p,k], :]
        More memory-friendly than the previous gather-loop
        Also return h and per-query stats [mean_d, std_d, effective_K, mean_phi] (B,P,4)
        """
        B, T, _, D = lat_proj.shape
        _, P, K    = top_idx.shape
        dev        = lat_proj.device

        w = torch.zeros(B, P, S, device=dev, dtype=lat_proj.dtype)
        w.scatter_(2, top_idx, weights_k)           # write K weights per query
    
        if CalRecVar == True:
            h = torch.einsum('btsd,bps->btpd', lat_proj, w)

            mask = valid_k.float()
            effective_K = mask.sum(dim=-1, keepdim=True)  # (B,P,1) number of valid sensors per query

            # Weighted mean/std for d_k and phi_k (using weights_k)
            mean_d = (d_k * weights_k * mask).sum(dim=-1, keepdim=True) / (effective_K + 1e-6)  
            var_d = ((d_k - mean_d)**2 * weights_k * mask).sum(dim=-1, keepdim=True) / (effective_K + 1e-6)
            std_d = torch.sqrt(var_d + 1e-6)  
            mean_phi = (phi_k * weights_k * mask).sum(dim=-1, keepdim=True) / (effective_K + 1e-6)  

            stats = torch.cat([mean_d, std_d, effective_K, mean_phi], dim=-1)  
            return h, stats
        else:
            return torch.einsum('btsd,bps->btpd', lat_proj, w)

    def forward(self,
                z: torch.Tensor,                # [B, T, S, D_raw] or [B, T, S+1, D_raw] if retain_cls=True
                Y: torch.Tensor,                # [B, P, 2/3]
                sensor_coords: torch.Tensor,    # [B, S, 2/3]
                mask: torch.Tensor,             # [B, T, S] (if retain_cls=True, this is for S+1; we slice below)
                phi_mean: torch.Tensor | None = None,

                padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:

        B, T, S_or_Sp1, D_raw = z.shape
        P              = Y.size(1)
        C              = self.N_channels
        dev            = z.device
        d_model        = self.lat_proj.out_features

        S = sensor_coords.size(1)  # true number of sensors (excludes CLS)

        if phi_mean is None: phi_mean = torch.ones(B, S, device=dev)  # [B,S]

        if self.per_sensor_sigma:
            if (self.log_sigma is None) or (self.log_sigma.numel() != S):
                self.log_sigma = nn.Parameter(torch.full((S,),
                                            math.log(self.bandwidth_init),
                                            device=dev))
            sigma = self.log_sigma.exp()  # (S,)
        else:
            sigma = torch.tensor(self.bandwidth_init, device=dev)

        # Combine mask and padding_mask; slice out CLS if present later
        effective_mask = mask
        if padding_mask is not None:
            padding_bt = padding_mask.unsqueeze(1).expand(-1, T, -1)  # [B,T,S or S+1]
            effective_mask = mask & padding_bt

        # Project tokens
        lat_proj = self.lat_proj(z)  # (B, T, S or S+1, d_model)

        if self.retain_cls:
            assert lat_proj.size(2) == S + 1, "retain_cls=True requires z to include CLS at index 0."
            cls_proj    = lat_proj[:, :, 0, :]      # (B,T,d)
            sensor_proj = lat_proj[:, :, 1:, :]     # (B,T,S,d)
            sensor_mask = effective_mask[:, :, 1:]  # (B,T,S)
        else:
            sensor_proj = lat_proj                  # (B,T,S,d)
            sensor_mask = effective_mask            # (B,T,S)

        # Positional tokens for queries
        coord_tok = self.coord_proj(self.pe(Y))     # (B,P,d)
        coord_tok = self.coord_norm(coord_tok)      # (B,P,d)
        coord_tok = coord_tok.unsqueeze(1).expand(B, T, P, d_model).reshape(B*T, P, d_model)  # (B*T,P,d)

        # Distance and importance scaling
        d = torch.cdist(Y, sensor_coords)  # (B,P,S)

        # Build a [B,S] validity mask for sensors (time-invariant proxy from t=0)
        sensor_valid_bs = sensor_mask[:, 0, :]  # [B,S] (True if valid at t=0)
        if padding_mask is not None and padding_mask.dim() == 2:
            sensor_valid_bs = sensor_valid_bs & padding_mask  # [B,S]

        # Set distances to inf for invalid sensors so they never get into top-k
        d = d.masked_fill(~sensor_valid_bs.unsqueeze(1), float('inf'))  # [B,P,S]
        phi   = phi_mean.detach()                                       # [B,S]
        gamma = self.importance_scale
        d_scaled = d / (phi[:, None, :] ** gamma + 1e-6)                # [B,P,S]

        # ------------------------------------------------------------
        # Unified top-k aggregation (K_eff = min(K, number_of_valid_sensors))
        # If self.top_k is None, K_eff = S (i.e., use all valid sensors)
        # ------------------------------------------------------------
        local_S = sensor_proj.size(2)  # == S
        if self.top_k is None:
            K_eff = local_S
        else:
            K_eff = min(self.top_k, local_S)

        # Top-k over smallest distances; if K_eff==S this is equivalent to "all sensors"
        # For rows where many sensors are invalid (inf), torch.topk returns the S finite ones first anyway.
        _, top_idx = torch.topk(d_scaled, K_eff, dim=2, largest=False)  # (B,P,K_eff)
        d_k = torch.gather(d_scaled, 2, top_idx)                        # (B,P,K_eff)

        # Gather phi and per-sensor sigma (if used) at the same indices
        phi_expan = phi[:, None, :].expand(-1, P, -1)                     # (B,P,S)
        phi_k     = torch.gather(phi_expan, 2, top_idx)                   # (B,P,K_eff)

        if self.per_sensor_sigma:
            sigma_k = sigma[top_idx]                                     # (B,P,K_eff)
        else:
            sigma_k = sigma                                              # scalar

        scores = -d_k / sigma_k                                          # (B,P,K_eff)
        scores -= scores.max(dim=-1, keepdim=True).values
        exp    = torch.exp(scores)

        # Validity at selected indices (time-invariant proxy from t=0)
        valid_k = torch.gather(sensor_valid_bs.unsqueeze(1).expand(-1, P, -1), 2, top_idx)  # (B,P,K_eff), bool
        exp = exp * valid_k.float()

        weights = exp / (exp.sum(dim=-1, keepdim=True) + 1e-6)  # (B,P,K_eff)

        # --- Combined Phi Incorporation (Weighted Fusion + Attention Modulation) ---
        # Weighted Fusion (multiply phi into weights, then re-normalize)
        if self.use_weighted_fusion:
            weights.mul_(phi_k * self.phi_scale)  # In-place multiplication
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)  # Re-normalize

        if self.CalRecVar:
            h, stats = self._topk_aggregate(sensor_proj, top_idx, weights,
                                            True, d_k, phi_k, valid_k, local_S)
        else:
            h = self._topk_aggregate(sensor_proj, top_idx, weights,
                                    False, d_k, phi_k, valid_k, local_S)
            stats = None

        h = self.agg_norm(h)              # (B,T,P,d)
        lat = h.reshape(B*T, P, d_model)  # (B*T,P,d)
        # ------------------------------------------------------------------------------

        if hasattr(self.cross_attn, 'to_v'):
            v_proj  = self.cross_attn.to_v
            out_proj = self.cross_attn.out
        else:
            mha = self.cross_attn.attn
            dim = mha.embed_dim
            v_weight = mha.in_proj_weight[2 * dim : 3 * dim]
            v_bias   = mha.in_proj_bias[2 * dim : 3 * dim] if mha.in_proj_bias is not None else None
            def v_proj(x): return F.linear(x, v_weight, v_bias)
            out_proj = mha.out_proj
        
        local_lat = out_proj(v_proj(lat))  # [B*T,P,d]
        local_pre = coord_tok + local_lat

        local_out_mean = None
        if self.retain_cls:
            cls_proj_bt = cls_proj.reshape(B*T, 1, d_model).expand(-1, P, -1)  # (B*T,P,d)
            fused_concat = torch.cat([local_pre, cls_proj_bt], dim=-1)  # (B*T, P, 2*d_model)
            fused_pre = self.fusion_proj(fused_concat)  # (B*T, P, d_model) - learned fusion

            fused_x = self.norm(fused_pre)
            fused_x = fused_x + self.mlp(fused_x)
            out_mean = self.head(fused_x).view(B, T, P, C)
            x_for_var = fused_x
        else:
            local_x = self.norm(local_pre)
            local_x = local_x + self.mlp(local_x)
            local_out_mean = self.head(local_x).view(B, T, P, C)
            out_mean = local_out_mean
            x_for_var = local_x

        # Optional variance head
        out_logvar = None
        if self.CalRecVar:
            stats_emb = self.stats_proj(stats)                     
            stats_emb = stats_emb.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, P, -1)
            x_var = torch.cat([x_for_var, stats_emb], dim=-1)      
            logvar = self.var_head(x_var)
            out_logvar = logvar.view(B, T, P, C)

        return local_out_mean, out_mean, out_logvar

    # Revised 0926: sensor reconstruction from sensor tokens (exclude CLS)
    def _forward(self,
                z: torch.Tensor,                # [B, T, S, D_raw] or [B, T, S+1, D_raw] if retain_cls=True
                Y: torch.Tensor,                # [B, P, 2/3]
                sensor_coords: torch.Tensor,    # [B, S, 2/3]
                mask: torch.Tensor,             # [B, T, S] (if retain_cls=True, this is for S+1; we slice below)
                phi_mean: torch.Tensor | None = None,

                padding_mask: torch.Tensor | None = None
                ):
        """
        Returns:
            global_out_mean: [B, T, S, C] if retain_cls=True else None
            out_mean:        [B, T, P, C]
            out_logvar:      [B, T, P, C] or None
        """
        B, T, S_or_Sp1, D_raw = z.shape
        P              = Y.size(1)
        C              = self.N_channels
        dev            = z.device
        d_model        = self.lat_proj.out_features

        S = sensor_coords.size(1)  # true number of sensors (excludes CLS)

        if phi_mean is None: 
            phi_mean = torch.ones(B, S, device=dev)  # [B,S]

        if self.per_sensor_sigma:
            if (self.log_sigma is None) or (self.log_sigma.numel() != S):
                self.log_sigma = nn.Parameter(torch.full((S,),
                                            math.log(self.bandwidth_init),
                                            device=dev))
            sigma = self.log_sigma.exp()  # (S,)
        else:
            sigma = torch.tensor(self.bandwidth_init, device=dev)

        # Combine mask and padding_mask; slice out CLS if present later
        effective_mask = mask
        if padding_mask is not None:
            padding_bt = padding_mask.unsqueeze(1).expand(-1, T, -1)  # [B,T,S or S+1]
            effective_mask = mask & padding_bt

        # Project tokens
        lat_proj = self.lat_proj(z)  # (B, T, S or S+1, d_model)

        if self.retain_cls:
            assert lat_proj.size(2) == S + 1, "retain_cls=True requires z to include CLS at index 0."
            cls_proj    = lat_proj[:, :, 0, :]      # (B,T,d)
            sensor_proj = lat_proj[:, :, 1:, :]     # (B,T,S,d)
            sensor_mask = effective_mask[:, :, 1:]  # (B,T,S)
        else:
            sensor_proj = lat_proj                  # (B,T,S,d)
            sensor_mask = effective_mask            # (B,T,S)

        # Positional tokens for queries (Y)
        coord_tok = self.coord_proj(self.pe(Y))     # (B,P,d)
        coord_tok = self.coord_norm(coord_tok)      # (B,P,d)
        coord_tok = coord_tok.unsqueeze(1).expand(B, T, P, d_model).reshape(B*T, P, d_model)  # (B*T,P,d)

        # Distance and importance scaling for Y aggregation
        d = torch.cdist(Y, sensor_coords)  # (B,P,S)

        # Build a [B,S] validity mask for sensors (time-invariant proxy from t=0)
        sensor_valid_bs = sensor_mask[:, 0, :]  # [B,S] (True if valid at t=0)
        if padding_mask is not None and padding_mask.dim() == 2:
            sensor_valid_bs = sensor_valid_bs & padding_mask  # [B,S]

        # Set distances to inf for invalid sensors so they never get into top-k
        d = d.masked_fill(~sensor_valid_bs.unsqueeze(1), float('inf'))  # [B,P,S]
        phi   = phi_mean.detach()                                       # [B,S]
        gamma = self.importance_scale
        d_scaled = d / (phi[:, None, :] ** gamma + 1e-6)                # [B,P,S]

        # ------------------------------------------------------------
        # Unified top-k aggregation (K_eff = min(K, number_of_valid_sensors))
        # If self.top_k is None, K_eff = S (i.e., use all valid sensors)
        # ------------------------------------------------------------
        local_S = sensor_proj.size(2)  # == S
        if self.top_k is None:
            K_eff = local_S
        else:
            K_eff = min(self.top_k, local_S)

        # Top-k over smallest distances
        _, top_idx = torch.topk(d_scaled, K_eff, dim=2, largest=False)  # (B,P,K_eff)
        d_k = torch.gather(d_scaled, 2, top_idx)                        # (B,P,K_eff)

        # Gather phi and per-sensor sigma (if used) at the same indices
        phi_expan = phi[:, None, :].expand(-1, P, -1)                   # (B,P,S)
        phi_k     = torch.gather(phi_expan, 2, top_idx)                 # (B,P,K_eff)

        if self.per_sensor_sigma:
            sigma_k = sigma[top_idx]                                    # (B,P,K_eff)
        else:
            sigma_k = sigma                                            # scalar

        scores = -d_k / sigma_k                                         # (B,P,K_eff)
        scores -= scores.max(dim=-1, keepdim=True).values
        exp    = torch.exp(scores)

        # Validity at selected indices (time-invariant proxy from t=0)
        valid_k = torch.gather(sensor_valid_bs.unsqueeze(1).expand(-1, P, -1), 2, top_idx)  # (B,P,K_eff), bool
        exp = exp * valid_k.float()

        weights = exp / (exp.sum(dim=-1, keepdim=True) + 1e-6)          # (B,P,K_eff)

        # --- Phi-weighted fusion (optional) ---
        if self.use_weighted_fusion:
            weights.mul_(phi_k * self.phi_scale)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Aggregate sensor latents for Y
        if self.CalRecVar:
            h, stats = self._topk_aggregate(sensor_proj, top_idx, weights,
                                            True, d_k, phi_k, valid_k, local_S)
        else:
            h = self._topk_aggregate(sensor_proj, top_idx, weights,
                                     False, d_k, phi_k, valid_k, local_S)
            stats = None

        h = self.agg_norm(h)              # (B,T,P,d)
        lat = h.reshape(B*T, P, d_model)  # (B*T,P,d)

        # Lightweight "V" -> "out" path from cross-attn module
        if hasattr(self.cross_attn, 'to_v'):
            v_proj  = self.cross_attn.to_v
            out_proj = self.cross_attn.out
        else:
            mha = self.cross_attn.attn
            dim = mha.embed_dim
            v_weight = mha.in_proj_weight[2 * dim : 3 * dim]
            v_bias   = mha.in_proj_bias[2 * dim : 3 * dim] if mha.in_proj_bias is not None else None
            def v_proj(x): return F.linear(x, v_weight, v_bias)
            out_proj = mha.out_proj

        # Y-field reconstruction
        local_lat = out_proj(v_proj(lat))  # [B*T,P,d]
        local_pre = coord_tok + local_lat

        if self.retain_cls:
            # Compute CLS projection once (on [B*T, 1, d]) and expand afterward
            cls_proj_bt = cls_proj.reshape(B*T, 1, d_model)  # [B*T, 1, d]
            cls_lat_single = out_proj(v_proj(cls_proj_bt))    # [B*T, 1, d] - compute once
            cls_lat = cls_lat_single.expand(-1, P, -1)        # [B*T, P, d] - expand result
            # cls_lat = self.norm(cls_lat)

            fused_pre = local_pre + cls_lat
            fused_x = self.norm(fused_pre)
            fused_x = fused_x + self.mlp(fused_x)
            out_mean = self.head(fused_x).view(B, T, P, C)
            x_for_var = fused_x
        else:
            local_x = self.norm(local_pre)
            local_x = local_x + self.mlp(local_x)
            out_mean = self.head(local_x).view(B, T, P, C)
            x_for_var = local_x

        # ------------------------------------------------------------
        # Sensor reconstruction from sensor tokens (exclude CLS)
        # Use the same v->out path and same head for consistency
        # ------------------------------------------------------------
        # Coordinate tokens for sensors
        coord_tok_s = self.coord_proj(self.pe(sensor_coords))               # (B,S,d)
        coord_tok_s = self.coord_norm(coord_tok_s)                          # (B,S,d)
        coord_tok_s_bt = coord_tok_s.unsqueeze(1).expand(B, T, S, d_model)\
                                    .reshape(B*T, S, d_model)               # (B*T,S,d)

        # Project sensor tokens through v->out
        sensor_bt = sensor_proj.reshape(B*T, S, d_model)                    # (B*T,S,d)
        sensor_lat_bt = out_proj(v_proj(sensor_bt))                         # (B*T,S,d)

        # Fuse with coordinates and predict
        global_pre_bt = coord_tok_s_bt + sensor_lat_bt                      # (B*T,S,d)

        if self.retain_cls:
            # Fuse CLS into sensor reconstruction
            cls_lat_s = cls_lat_single.expand(-1, S, -1)                    # [B*T, S, d] - expand the same projected CLS to S
            # cls_lat_s = self.norm(cls_lat_s)
            global_pre_bt = global_pre_bt + cls_lat_s                       # Add CLS to the sensor pre features

        global_x_bt = self.norm(global_pre_bt)
        global_x_bt = global_x_bt + self.mlp(global_x_bt)
        global_out_mean = self.head(global_x_bt).view(B, T, S, C)           # (B,T,S,C)

        # Optionally zero-out invalid sensor positions
        if sensor_mask is not None:
            global_out_mean = global_out_mean.masked_fill(
                (~sensor_mask).unsqueeze(-1), 0.0
            )

        # Optional variance head on Y
        out_logvar = None
        if self.CalRecVar:
            stats_emb = self.stats_proj(stats)                     
            stats_emb = stats_emb.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, P, -1)
            x_var = torch.cat([x_for_var, stats_emb], dim=-1)      
            logvar = self.var_head(x_var)
            out_logvar = logvar.view(B, T, P, C)

        return global_out_mean, out_mean, out_logvar

# ==============================
# Complete Model wrappers
# ==============================
class TD_ROM(nn.Module):
    def __init__(self,
                 fieldencoder: nn.Module,
                 temporaldecoder: nn.Module,
                 fielddecoder: nn.Module,

                 delta_t: float,
                 N_window: int,
                 CheckPhi: bool = False,
                 stage: int = -1,
                 ):

        super().__init__()
        self.stage                  = stage

        self.fieldencoder           = fieldencoder
        self.temporaldecoder        = temporaldecoder
        self.TemporalDecoderAdapter = TemporalDecoderAdapter(temporaldecoder)
        self.decoder                = fielddecoder

        self.delta_t                = delta_t
        self.N_window               = N_window
        self.CheckPhi               = CheckPhi

    def forward(self,
                G_down: torch.Tensor,     # [B, N_ts, N_xs, F=N_dim+N_c+1]  (down-sampled)
                G_full: torch.Tensor,     # [B, T_full, N_pts, N_c]
                Y     : torch.Tensor,     # [N_pts, N_dim] 
                U     : torch.Tensor,     # [B, N-para] (can be dummy tensor)
                teacher_force_prob: float,
                ):    
        """
        Returns:
        G_u          : reconstructed physical field  [B, T_full, N_points, N_c]
        latent_traj  : latent trajectory F(t)        [B, T_full, F_dim]
        G_obs       : latent tokens of the observed prefix
        """ 

        B, T_full, N_pts, _ = G_full.shape

        # --- Transformer -> latent tokens of ALL observed frames ---
        G_obs  = self.fieldencoder(G_down, U)

        is_multi_token = (G_obs.dim() == 4)
        if is_multi_token:
            B, Tobs, L, D = G_obs.shape
            latent_seed = G_obs.permute(0, 2, 1, 3).contiguous().view(B * L, Tobs, D)  # (B*L, Tobs, D)
        else:
            latent_seed = G_obs  # (B, Tobs, D)

        if self.stage == 0 or self.N_window == T_full:
            latent_traj = G_obs
        else:
            # --- Integrate the latent dynamics ---------------------------------
            if self.training:
                # choose parallel teacher-forced *or* autoregressive schedule
                if teacher_force_prob == 1.0:          # plain parallel
                    latent_traj = self.temporaldecoder(x_seq = latent_seed)
                else:                                  # scheduled sampling
                    output = self.TemporalDecoderAdapter.forward_autoreg(
                        G_latent           = latent_seed,
                        N_Fore             = T_full,
                        N_window           = self.N_window,
                        teacher_force_seq  = None,
                        teacher_force_prob = teacher_force_prob,
                        truncate_k         = 64,               
                    )
            else: 
                output = self.TemporalDecoderAdapter(
                    G_latent = latent_seed,
                    N_Fore   = T_full,
                    N_window = self.N_window)
            latent_traj, latent_traj_logvar = output

        if is_multi_token:
            latent_traj = latent_traj.view(B, L, T_full, D).permute(0, 2, 1, 3)  # (B, T_full, L, D)

        # --- Decode back to physical space ---------------------------------
        G_u     = self.decoder(latent_traj, Y)             # [B, T_full, N_pts, 1]

        G_u_cls = None
        G_u_logvar = None
        traj_logvar = None
        G_u_mean_Sens = None
        G_u_logvar_Sens = None

        return G_u, G_u_logvar, G_obs, latent_traj, traj_logvar, G_u_cls, G_u_mean_Sens,G_u_logvar_Sens

# Domain decomposition (DD) version of model wrapper
class TD_ROM_Bay_DD(nn.Module):
    def __init__(self,
                 cfg: dict,
                 fieldencoder: nn.Module,
                 temporaldecoder: nn.Module,
                 fielddecoder: nn.Module,

                 delta_t: float,
                 N_window: int,
                 CheckPhi: bool = False,
                 stage: int = -1,

                 use_adaptive_selection: bool  = False,
                 CalRecVar             : bool  = False, 
                 retain_cls            : bool  = False, 
                 Use_imp_in_dyn        : bool  = False,
                 ):

        super().__init__()
        self.cfg                    = cfg
        self.stage                  = stage

        self.fieldencoder           = fieldencoder
        self.temporaldecoder        = temporaldecoder
        self.TemporalDecoderAdapter = TemporalDecoderAdapter(temporaldecoder)
        self.decoder                = fielddecoder
        self.retain_cls             = retain_cls
        self.Use_imp_in_dyn         = Use_imp_in_dyn

        self.delta_t                = delta_t
        self.N_window               = N_window
        self.CheckPhi               = CheckPhi
        
        self.Supervise_Sensors      = cfg.get('Supervise_Sensors', False)

        self.CalRecVar              = CalRecVar
        self.use_adaptive_selection = use_adaptive_selection
        if self.use_adaptive_selection:

            # MLP-1: spatial uncertainty
            self.phi_mlp_1 = nn.Sequential(
                nn.Linear(2, cfg["bayesian_phi"]["phi_mlp_hidden_dim"]), nn.ReLU(),  # Input: (x,y)
                nn.Linear(cfg["bayesian_phi"]["phi_mlp_hidden_dim"], cfg["bayesian_phi"]["phi_mlp_hidden_dim"]), nn.ReLU(),
                nn.Linear(cfg["bayesian_phi"]["phi_mlp_hidden_dim"], 2)  # Output: log(alpha), log(beta) for positivity/stability
            )
            # Initialize with small random weights 
            torch.nn.init.xavier_uniform_(self.phi_mlp_1[0].weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.phi_mlp_1[2].weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.phi_mlp_1[4].weight)

            # MLP-2: temporal uncertainty
            self.phi_mlp_2 = nn.Sequential(
                nn.Linear(2, cfg["bayesian_phi"]["phi_mlp_hidden_dim"]), nn.ReLU(),  
                nn.Linear(cfg["bayesian_phi"]["phi_mlp_hidden_dim"], cfg["bayesian_phi"]["phi_mlp_hidden_dim"]), nn.ReLU(),
                nn.Linear(cfg["bayesian_phi"]["phi_mlp_hidden_dim"], 2)
            )
            torch.nn.init.xavier_uniform_(self.phi_mlp_2[0].weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.phi_mlp_2[2].weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.phi_mlp_2[4].weight)

            self.avg_residuals = None  # EMA buffer for per-point reconstruction uncertainty [N_pts, 1]

    def forward(self,
                G_down: torch.Tensor,     # [B, N_ts, N_xs, F]  (down-sampled)
                G_full: torch.Tensor,     # [B, T_full, N_pts, 1]
                Y     : torch.Tensor,     # [N_pts, N_dim] 
                U     : torch.Tensor,     # [B, N-para] (can be dummy tensor)
                teacher_force_prob: float,
                ):    

        # --- Extract number of required integration steps ---
        
        B, T_full, N_pts, _ = G_full.shape
        _, N_ts, N_xs, F_feat = G_down.shape

        # Compute original phi on retained sensors if adaptive
        original_phi = None
        if self.use_adaptive_selection:
            current_coords = G_down[:, 0, :, :2]  # [B, K, 2] (per-batch, t=0; shared across t for each case in a batch)
            assert current_coords.shape == (B, N_xs, 2), f"Coords shape mismatch: {current_coords.shape}"

            log_ab_1 = self.phi_mlp_1(current_coords)                         
            alpha_1  = torch.exp(log_ab_1[:, :, 0]) + 1e-3                     
            beta_1   = torch.exp(log_ab_1[:, :, 1]) + 1e-3
            if self.training: phi_1 = torch.distributions.Beta(alpha_1, beta_1).rsample()
            else:
                # Compute mean phi (Beta expectation) instead of sampling
                mean_phi_1 = torch.clamp(alpha_1 / (alpha_1 + beta_1) , min=1e-3, max=1-1e-3)  # Clamp for stability
                phi_1 = mean_phi_1

            # Temporal contributions by phi_mlp_2:
            if self.stage == 1 and self.cfg["bayesian_phi"]["update_in_stage1"] == True:
                log_ab_2 = self.phi_mlp_2(current_coords)                        
                alpha_2  = torch.exp(log_ab_2[:, :, 0]) + 1e-3                     
                beta_2   = torch.exp(log_ab_2[:, :, 1]) + 1e-3
                if self.training: phi_2 = torch.distributions.Beta(alpha_2, beta_2).rsample()
                else:
                    mean_phi_2 = torch.clamp(alpha_2 / (alpha_2 + beta_2), min=1e-3, max=1-1e-3)
                    phi_2 = mean_phi_2
            else:
                phi_2 = torch.ones_like(phi_1) # No temporal uncertainty considered

            original_phi = phi_1 * phi_2

        # --- Transformer encoder -> latent tokens of ALL observed frames ---
        G_obs, mask_from_encoder, sensor_coords_from_encoder, merged_phi = self.fieldencoder(G_down, U, original_phi) 
        is_multi_token = (G_obs.dim() == 4)

        if is_multi_token:
            B, Tobs, L, D = G_obs.shape
            if self.cfg.get('decoder_type', "CausalTrans") != "UD_Trans":
                latent_seed = G_obs.permute(0, 2, 1, 3).contiguous().view(B * L, Tobs, D)
            else:
                latent_seed = G_obs 

        if self.stage == 0 or self.N_window == T_full: 
            latent_traj = G_obs
            latent_traj_logvar = None
        else:
            # --- Integrate the latent dynamics ---------------------------------

            imp = original_phi.detach() if self.Use_imp_in_dyn is True else None

            if self.training:
                output = self.TemporalDecoderAdapter.forward_autoreg(
                    G_latent           = latent_seed,
                    N_Fore             = T_full,
                    imp                = imp,
                    N_window           = self.N_window,
                    teacher_force_seq  = None,      # ground truth tokens
                    teacher_force_prob = teacher_force_prob,
                    truncate_k         = 64,               
                )
            else: # evaluation / inference
                output = self.TemporalDecoderAdapter(
                    G_latent = latent_seed,
                    N_Fore   = T_full,
                    N_window = self.N_window,
                    imp      = imp,
                    )
            latent_traj, latent_traj_logvar = output

        if is_multi_token and self.cfg.get('decoder_type', "CausalTrans") != "UD_Trans":
            latent_traj = latent_traj.view(B, L, T_full, D).permute(0, 2, 1, 3)  # (B, T_full, L, D)
            if latent_traj_logvar is not None:
                    latent_traj_logvar = latent_traj_logvar.view(B, L, T_full, D).permute(0, 2, 1, 3)

        # --- Decode back to physical space ---------------------------------
        phi_mean = merged_phi if self.use_adaptive_selection else None
        self.phi_mean_ = phi_mean
        self.sensor_coords_ = sensor_coords_from_encoder

        G_u_mean_Sens = G_u_logvar_Sens = None
        G_u_mean = G_u_logvar = None

        G_u_cls, G_u_mean, G_u_logvar = self.decoder(latent_traj, Y, sensor_coords=sensor_coords_from_encoder, 
                mask=mask_from_encoder[:, -1:] if mask_from_encoder.dim() > 1 else mask_from_encoder, phi_mean=phi_mean) 
        
        if self.Supervise_Sensors:  # Only decode the values at the sensors's locations
            G_u_cls_sens, G_u_mean_Sens, G_u_logvar_Sens = self.decoder(latent_traj, sensor_coords_from_encoder, sensor_coords=sensor_coords_from_encoder, 
                    mask=mask_from_encoder[:, -1:] if mask_from_encoder.dim() > 1 else mask_from_encoder, phi_mean=phi_mean)

        return (G_u_mean, G_u_logvar, 
                G_obs, latent_traj, latent_traj_logvar, 
                G_u_cls, 
                G_u_mean_Sens, G_u_logvar_Sens)

    #  Revised 0926, reconstructor will rebuild sensor values
    def _forward(self,
                G_down: torch.Tensor,     # [B, N_ts, N_xs, F]  (down-sampled)
                G_full: torch.Tensor,     # [B, T_full, N_pts, 1]
                Y     : torch.Tensor,     # [N_pts, N_dim] 
                U     : torch.Tensor,     # [B, N-para] (can be dummy tensor)
                teacher_force_prob: float,
                ):    

        # --- Extract number of required integration steps ---
        
        B, T_full, N_pts, _ = G_full.shape
        _, N_ts, N_xs, F_feat = G_down.shape

        # Compute original phi on retained sensors if adaptive
        original_phi = None
        if self.use_adaptive_selection:
            current_coords = G_down[:, 0, :, :2]  # [B, K, 2] (per-batch, t=0; shared across t for each case in a batch)
            assert current_coords.shape == (B, N_xs, 2), f"Coords shape mismatch: {current_coords.shape}"

            log_ab_1 = self.phi_mlp_1(current_coords)                         
            alpha_1  = torch.exp(log_ab_1[:, :, 0]) + 1e-3                     
            beta_1   = torch.exp(log_ab_1[:, :, 1]) + 1e-3
            if self.training: phi_1 = torch.distributions.Beta(alpha_1, beta_1).rsample()
            else:
                # Compute mean phi (Beta expectation) instead of sampling
                mean_phi_1 = torch.clamp(alpha_1 / (alpha_1 + beta_1) , min=1e-3, max=1-1e-3)  # Clamp for stability
                phi_1 = mean_phi_1

            # Temporal contributions by phi_mlp_2:
            if self.stage == 1 and self.cfg["bayesian_phi"]["update_in_stage1"] == True:
                log_ab_2 = self.phi_mlp_2(current_coords)                        
                alpha_2  = torch.exp(log_ab_2[:, :, 0]) + 1e-3                     
                beta_2   = torch.exp(log_ab_2[:, :, 1]) + 1e-3
                if self.training: phi_2 = torch.distributions.Beta(alpha_2, beta_2).rsample()
                else:
                    mean_phi_2 = torch.clamp(alpha_2 / (alpha_2 + beta_2), min=1e-3, max=1-1e-3)
                    phi_2 = mean_phi_2
            else:
                phi_2 = torch.ones_like(phi_1) # No temporal uncertainty considered

            original_phi = phi_1 * phi_2

        # --- Transformer encoder -> latent tokens of ALL observed frames ---
        G_obs, mask_from_encoder, sensor_coords_from_encoder, merged_phi = self.fieldencoder(G_down, U, original_phi) 
        is_multi_token = (G_obs.dim() == 4)

        if is_multi_token:
            B, Tobs, L, D = G_obs.shape
            if self.cfg.get('decoder_type', "CausalTrans") != "UD_Trans":
                latent_seed = G_obs.permute(0, 2, 1, 3).contiguous().view(B * L, Tobs, D)
            else:
                latent_seed = G_obs 

        if self.stage == 0 or self.N_window == T_full: 
            latent_traj = G_obs
            latent_traj_logvar = None
        else:
            # --- Integrate the latent dynamics ---------------------------------

            imp = original_phi.detach() if self.Use_imp_in_dyn is True else None

            if self.training:
                output = self.TemporalDecoderAdapter.forward_autoreg(
                    G_latent           = latent_seed,
                    N_Fore             = T_full,
                    imp                = imp,
                    N_window           = self.N_window,
                    teacher_force_seq  = None,      # ground truth tokens
                    teacher_force_prob = teacher_force_prob,
                    truncate_k         = 64,               
                )
            else: # evaluation / inference
                output = self.TemporalDecoderAdapter(
                    G_latent = latent_seed,
                    N_Fore   = T_full,
                    N_window = self.N_window,
                    imp      = imp,
                    )
            latent_traj, latent_traj_logvar = output

        if is_multi_token and self.cfg.get('decoder_type', "CausalTrans") != "UD_Trans":
            latent_traj = latent_traj.view(B, L, T_full, D).permute(0, 2, 1, 3)  # (B, T_full, L, D)
            if latent_traj_logvar is not None:
                    latent_traj_logvar = latent_traj_logvar.view(B, L, T_full, D).permute(0, 2, 1, 3)

        # --- Decode back to physical space ---------------------------------
        phi_mean = merged_phi if self.use_adaptive_selection else None
        self.phi_mean_ = phi_mean
        self.sensor_coords_ = sensor_coords_from_encoder

        G_u_mean_Sens = G_u_logvar_Sens = None
        G_u_mean = G_u_logvar = None

        G_u_cls, G_u_mean, G_u_logvar = self.decoder(latent_traj, Y, sensor_coords=sensor_coords_from_encoder, 
                mask=mask_from_encoder[:, -1:] if mask_from_encoder.dim() > 1 else mask_from_encoder, phi_mean=phi_mean) 
        # G_u_cls is the sensor's values rebuilt on CLS
        
        if self.Supervise_Sensors:  # Only decode the values at the sensors's locations
            G_u_mean_Sens = G_u_cls

        return (G_u_mean, G_u_logvar, 
                G_obs, latent_traj, latent_traj_logvar, 
                G_u_cls, 
                G_u_mean_Sens, G_u_logvar_Sens)

#------------------------------------------
# These parts are for Senseriver from https://doi.org/10.1038/s42256-023-00746-x
#------------------------------------------

class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def mlp_Sen(num_channels: int):
    return Sequential(
                        nn.LayerNorm(num_channels),
                        nn.Linear(num_channels, num_channels),
                        nn.GELU(),
                        nn.Linear(num_channels, num_channels),
                    )


def cross_attention_layer_Sen( num_q_channels: int, 
                           num_kv_channels: int, 
                           num_heads: int, 
                           dropout: float, 
                           activation_checkpoint: bool = False):
    
    layer = Sequential(
        Residual_Sen(CrossAttention_Sen(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual_Sen(mlp_Sen(num_q_channels), dropout),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_layer_Sen(num_channels: int, 
                         num_heads: int, 
                         dropout: float, 
                         activation_checkpoint: bool = False):
    
    layer = Sequential(
        Residual_Sen(SelfAttention_Sen(num_channels, num_heads, dropout), dropout), 
        Residual_Sen(mlp_Sen(num_channels), dropout)
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_block_Sen(num_layers: int, 
                         num_channels: int, 
                         num_heads: int, 
                         dropout: float, 
                         activation_checkpoint: bool = False
                        ):
    
    layers = [self_attention_layer_Sen(
                            num_channels, 
                            num_heads, 
                            dropout, 
                            activation_checkpoint) for _ in range(num_layers)]
    
    return Sequential(*layers)


class Residual_Sen(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention_Sen(nn.Module):
    
    def __init__(self, num_q_channels: int, 
                 num_kv_channels: int, 
                 num_heads: int, 
                 dropout: float):
        
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


class CrossAttention_Sen(nn.Module):
    
    def __init__(self, 
                 num_q_channels: int, 
                 num_kv_channels: int, 
                 num_heads: int, 
                 dropout: float):
        
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention_Sen(
                                            num_q_channels=num_q_channels, 
                                            num_kv_channels=num_kv_channels, 
                                            num_heads=num_heads, 
                                            dropout=dropout
                                            )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)

class SelfAttention_Sen(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention_Sen(
                                            num_q_channels=num_channels, 
                                            num_kv_channels=num_channels, 
                                            num_heads=num_heads, 
                                            dropout=dropout
                                            )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)

def _fourier_encode(coords, *, num_bands: int, sizes: tuple[int, ...], max_freq=1.0, include_orig=False):
    """
    coords: [..., d] in [-1, 1], d == len(sizes)
    sizes: spatial sizes per dim, e.g., (H, W) or (D, H, W)
    returns: [..., 2 * d * num_bands] with sin/cos only (no raw coords)
    """
    *batch, d = coords.shape
    assert d == len(sizes)
    parts = []
    for i, size_i in enumerate(sizes):
        freqs = torch.linspace(1.0, size_i / 2.0, num_bands, device=coords.device)  # [B]
        xi = coords[..., i:i+1]                                                     # [..., 1]
        arg = xi * freqs.view(*([1] * (coords.dim() - 1)), -1) * math.pi            # [..., B]
        parts.append(arg.sin())
        parts.append(arg.cos())
    return torch.cat(parts, dim=-1)  # [..., 2 * d * num_bands]

class Encoder_Sen(nn.Module):
    def __init__(self,
                 input_ch: int,                 # sensor–value dimension
                 coord_dim: int,                # 2 for (x,y) or 3 for (x,y,z)
                 domain_sizes: tuple,
                 preproc_ch: int | None = 128,
                 num_latents: int = 16,
                 num_latent_channels: int = 64,
                 num_layers: int = 3,
                 num_cross_attention_heads: int = 4,
                 num_self_attention_heads: int = 4,
                 num_self_attention_layers_per_block: int = 6,
                 num_pos_bands: int = 32,
                 max_pos_freq: float = 1.0,
                 dropout: float = 0.0,
                 activation_checkpoint: bool = False):
        super().__init__()

        self.num_layers     = num_layers
        self.input_ch       = input_ch
        self.num_pos_bands  = num_pos_bands
        self.max_pos_freq   = max_pos_freq
        self.coord_dim      = coord_dim

        self.pos_dim        = coord_dim * (2 * num_pos_bands)
        self.domain_sizes   = domain_sizes

        total_in = self.input_ch + self.pos_dim
        self.preproc = nn.Linear(total_in, preproc_ch) if preproc_ch else None
        if not preproc_ch:
            preproc_ch = total_in

        def create_layer():
            return Sequential(
                cross_attention_layer_Sen(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=preproc_ch,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint),
                self_attention_block_Sen(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint),
            )

        self.layer_1 = create_layer()
        if num_layers > 1:
            self.layer_n = create_layer()

        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    # ------------------------------------------------------------
    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    # ------------------------------------------------------------
    def forward(self,
                G_down: torch.Tensor, 
                U     : torch.Tensor, 
                pad_mask=None):
        B, T, N_s, _ = G_down.shape

        sensor_pos = G_down[..., :2]           
        sensor_val = G_down[..., 2: 2 + self.input_ch]

        # Fourier-encode positions and concatenate -----------------------
        pos_feat = _fourier_encode(sensor_pos,
                                  num_bands=self.num_pos_bands,
                                  sizes = self.domain_sizes,
                                  max_freq=self.max_pos_freq,)         # [B,T,N_s,pos_dim]
        x = torch.cat([sensor_val, pos_feat], dim=-1)                  # [B,T,N_s,tot_in]
        x = x.view(B*T, N_s, -1)                                       # [B*T,N_s,tot_in]

        if self.preproc:
            x = self.preproc(x)

        # ---------- Perceiver core --------------------------------------
        lat = repeat(self.latent, "... -> b ...", b=B*T)               # [B*T,N_lat,D]
        lat = self.layer_1(lat, x, pad_mask)
        for _ in range(self.num_layers - 1):
            lat = self.layer_n(lat, x, pad_mask)

        lat = lat.view(B, T, *lat.shape[1:])                           # [B,T,N_lat,D]
        return lat

class Decoder_Sen(nn.Module):
    """
    Spatial decoder that generates Fourier features internally
    from raw coordinates `Y` (range expected in [-1,1]).
    """
    def __init__(self, *,
                 coord_dim: int,
                 domain_sizes: tuple,
                 preproc_ch: int | None,
                 num_pos_bands: int = 64,
                 max_pos_freq: float = 1.0,
                 num_latent_channels: int,
                 latent_size: int,
                 num_cross_attention_heads: int,
                 num_output_channels: int,
                 dropout: float = 0.0,
                 activation_checkpoint: bool = False,
                 ):
        super().__init__()

        self.num_pos_bands = num_pos_bands
        self.max_pos_freq  = max_pos_freq
        self.domain_sizes   = domain_sizes
        self.include_orig  = False     
        
        pos_dim = coord_dim * (self.include_orig + 2 * num_pos_bands)
        q_chan  = pos_dim + num_latent_channels

        q_in = preproc_ch if preproc_ch else q_chan
        self.postproc = nn.Linear(q_in, num_output_channels)

        if preproc_ch:
            self.preproc = nn.Linear(q_chan, preproc_ch)
        else:
            self.preproc = None

        self.cross_attention = cross_attention_layer_Sen(
            num_q_channels=q_in,
            num_kv_channels=num_latent_channels,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
            activation_checkpoint=activation_checkpoint)

        self.output = nn.Parameter(torch.empty(latent_size, num_latent_channels))
        self._init_parameters()

    # ------------------------------------------------------------------
    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, latents: torch.Tensor, coords_raw: torch.Tensor):
        """
        latents   : [B, T, N_lat, D]  after TD_ROM reshape
        coords_raw: [B, N_pts, coord_dim]  (with batch dim)
        """

        if latents.dim() == 3:
            latents = latents.unsqueeze(2)  # [B,T,1,D]
        B, T, N_lat, D = latents.shape
        Bc, N_pts, coord_dim = coords_raw.shape
        assert B == Bc, "coords_raw should have the same batch size as latents"

        coords_feat = _fourier_encode(
            coords_raw,
            num_bands=self.num_pos_bands,
            sizes = self.domain_sizes,
            max_freq=self.max_pos_freq,
            include_orig=self.include_orig
        )  # [B, N_pts, pos_dim]

        # broadcast across time only
        coords_feat = coords_feat.unsqueeze(1).expand(B, T, N_pts, -1).reshape(B * T, N_pts, -1)

        latents_flat = latents.reshape(B * T, N_lat, D)
        out_token = repeat(self.output, "... -> b ...", b=B * T)       # [B*T,latent_size,D]
        out_token = torch.repeat_interleave(out_token, N_pts, dim=1)   # [B*T,N_pts*latent_size,D]
        q = torch.cat([coords_feat, out_token], dim=-1)

        if self.preproc:
            q = self.preproc(q)
        q = self.cross_attention(q, latents_flat)
        q = self.postproc(q)
        return q.view(B, T, N_pts, -1)

class TDROMEncoder_Sen(Encoder_Sen):

    def __init__(self, *args, latent_to_feat_dim: int, **kw):
        super().__init__(*args, **kw)
        self.to_mu     = nn.Linear(self.latent.shape[-1], latent_to_feat_dim)
        self.to_logvar = nn.Linear(self.latent.shape[-1], latent_to_feat_dim)

    def forward(self,
                G_Down: torch.Tensor, 
                pad_mask=None):
        lat = super().forward(G_Down, pad_mask)        # [B,T,N_lat,D]

        # collapse latent tokens → a single feature per time-step -------
        G_obs = lat.mean(dim=2)                                        # [B,T,D]

        return G_obs

#------------------------------------------
# These parts are for beta-VAE from https://doi.org/10.1038/s41467-024-45323-x
# & MLP-CNN sparse reconstructor from https://doi.org/10.1038/s42256-025-01063-1
#------------------------------------------

# Modules for CNN-based beta-VAE
def get_act(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "elu" : nn.ELU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU()
    }[name.lower()]

class VAE_Encoder(nn.Module):
    """
    Generic conv encoder that infers its own output size.
    """
    def __init__(self, cfg: dict, input_ch: int):
        super().__init__()
        self.cfg = cfg
        act = get_act(cfg["activation"])
        layers = []
        in_ch = input_ch
        for out_ch in cfg["enc_channels"]:
            layers += [
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=cfg["kernel_size"],
                          stride=cfg["stride"],
                          padding=cfg["padding"]),
                act
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # infer flat dimension with one dummy pass
        with torch.no_grad():
            H, W = cfg["Num_y"], cfg["Num_x"]
            dummy = torch.zeros(1, input_ch, H, W)
            flat_dim = self.conv(dummy).numel()

        self.fc_mu      = nn.Linear(flat_dim, cfg["latent_dim"])
        self.fc_logvar  = nn.Linear(flat_dim, cfg["latent_dim"])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x).flatten(1) 
        return self.fc_mu(x), self.fc_logvar(x)

class ConstantCrop(nn.Module):
    """
    Wraps ConstantPad2d with negative padding to crop to the desired (H, W).
    If no crop needed, acts as identity.
    Crops from the bottom and right by default (matches original code behavior).
    """
    def __init__(self, crop_h: int = 0, crop_w: int = 0):
        super().__init__()
        # ConstantPad2d pads in order (left, right, top, bottom). Negative = crop.
        self.pad = None
        if crop_h > 0 or crop_w > 0:
            self.pad = nn.ConstantPad2d((0, -crop_w, 0, -crop_h), 0.0)

    def forward(self, x):
        if self.pad is None:
            return x
        return self.pad(x)

class VAE_Decoder(nn.Module):
    """
    Mirrors Encoder. Uses ConvTranspose2d and computes per-stage crops automatically
    via ConstantPad2d to hit exact target size (Num_y, Num_x).
    """
    def __init__(self, cfg: dict, output_ch: int):
        super().__init__()
        self.cfg = cfg
        self.output_ch = output_ch
        act = get_act(cfg["activation"])

        H, W = cfg["Num_y"], cfg["Num_x"]

        # Probe encoder to get starting feature map (C0, h0, w0)
        enc_dummy = VAE_Encoder(cfg, input_ch=output_ch).conv  # output_ch == input_ch in your build_model
        with torch.no_grad():
            dummy = torch.zeros(1, output_ch, H, W)
            feat = enc_dummy(dummy)
            C0, h0, w0 = feat.shape[1:]
            flat_dim = C0 * h0 * w0

        # FC part
        self.fc = nn.Sequential(
            nn.Linear(cfg["latent_dim"], cfg["linear_hidden"]), act,
            nn.Linear(cfg["linear_hidden"], flat_dim), act,
            nn.Unflatten(1, (C0, h0, w0))
        )

        # Deconv part: build stages and compute crops
        channels = list(cfg["dec_channels"])
        assert channels[0] == C0, "First dec_channels entry must equal encoder's last out_channels"
        stride = cfg["stride"]
        k = cfg["kernel_size"]
        p = cfg["padding"]
        # For k=3, s=2, p=1, op=1, each stage doubles exactly: out = 2*in
        op = stride - 1

        deconv_layers = []
        crop_layers = []
        in_ch = channels[0]
        cur_h, cur_w = h0, w0
        remaining_stages = len(channels)  # including the final output layer we'll add later

        for out_ch in channels[1:]:
            # One deconv stage
            deconv_layers += [
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=k, stride=stride, padding=p,
                    output_padding=op
                ),
                act
            ]
            # Update current size (formula for convtranspose)
            cur_h = (cur_h - 1) * stride - 2 * p + k + op
            cur_w = (cur_w - 1) * stride - 2 * p + k + op

            remaining_stages -= 1  
            # Total upsample factor remaining after current stage:
            scale_left = (stride ** (remaining_stages))  # since final layer also uses same stride

            target_h_now = (H + scale_left - 1) // scale_left
            target_w_now = (W + scale_left - 1) // scale_left

            crop_h = max(cur_h - target_h_now, 0)
            crop_w = max(cur_w - target_w_now, 0)

            crop_layers.append(ConstantCrop(crop_h, crop_w))
            # Apply the crop to our tracked sizes
            cur_h -= crop_h
            cur_w -= crop_w
            in_ch = out_ch

        # Final output layer to output_ch
        deconv_layers += [
            nn.ConvTranspose2d(
                in_ch, output_ch,
                kernel_size=k, stride=stride, padding=p,
                output_padding=op
            )
        ]
        # Update size once more
        cur_h = (cur_h - 1) * stride - 2 * p + k + op
        cur_w = (cur_w - 1) * stride - 2 * p + k + op

        # Final crop to (H, W) if needed
        final_crop_h = max(cur_h - H, 0)
        final_crop_w = max(cur_w - W, 0)
        final_crop = ConstantCrop(final_crop_h, final_crop_w)

        self.deconvs = nn.ModuleList()
        self.crops = nn.ModuleList()

        idx = 0
        for i in range(len(channels) - 1):
            self.deconvs.append(nn.Sequential(deconv_layers[idx], deconv_layers[idx + 1]))
            idx += 2
            self.crops.append(crop_layers[i])
        # Append the final output layer (no activation)
        self.final_deconv = deconv_layers[-1]
        self.final_crop = final_crop

        self.act = act

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)

        for deconv, crop in zip(self.deconvs, self.crops):
            x = deconv(x)
            x = crop(x)
        x = self.final_deconv(x)
        x = self.final_crop(x)

        return x

# Modules for MLP-CNN-Reconstructor
class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_size: Tuple[int, int]):
        super().__init__()
        self.output_size = output_size  # (H, W) for the output grid
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size[0] * output_size[1]))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        x: [B, N_obs * (2 + C)] flattened sparse observations
        grid: [B, 1, H, W] embedded representation
        """
        B = x.shape[0]
        out = self.mlp(x)
        return out.view(B, 1, self.output_size[0], self.output_size[1])

class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_bn=False, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels) if use_bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class ReconstructorBlock(nn.Module):
    """Reconstructor block with upsampling"""
    def __init__(self, in_channels, middle_channels, out_channels, use_bn=False, dropout=0.0):
        super().__init__()
        self.decode = nn.Sequential(
            ConvBlock(in_channels, middle_channels, use_bn=use_bn, dropout=dropout),
            ConvBlock(middle_channels, middle_channels, use_bn=use_bn, dropout=0.0),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        return self.decode(x)

class SparseCNNReconstructor(nn.Module):
    """
    Takes sparse observations and reconstructs full field using MLP embedding + CNN.
    """
    def __init__(self, 
                 n_channels: int,
                 N_obs: int,
                 output_size: Tuple[int, int],              # (Num_y, Num_x)
                 embedding_size: Tuple[int, int] = (8, 8),  # Size after MLP embedding
                 hidden_dims: list = [128, 256],
                 conv_channels: list = [32, 64, 64, 64, 64],
                 use_bn: bool = False,
                 dropout: float = 0.0):

        super().__init__()
        
        self.n_channels = n_channels
        self.N_obs = N_obs
        self.output_size = output_size
        self.embedding_size = embedding_size
        
        # MLP embedding: takes flattened sparse input and produces grid
        input_dim = N_obs * (2 + n_channels)  # coordinates + values
        self.mlp_embedding = MLPEmbedding(input_dim, hidden_dims, embedding_size)
        
        self.channel_embedding = nn.Conv2d(1, conv_channels[0], kernel_size=1)
        self.decoders = nn.ModuleList()
        
        # Calculate number of upsampling stages needed
        h_scale = output_size[0] / embedding_size[0]
        w_scale = output_size[1] / embedding_size[1]
        n_upsample = int(np.log2(max(h_scale, w_scale)))
        
        in_ch = conv_channels[0]
        for i in range(n_upsample):
            out_ch = conv_channels[min(i+1, len(conv_channels)-1)]
            self.decoders.append(
                ReconstructorBlock(in_ch, out_ch, out_ch, use_bn=use_bn, dropout=dropout if i < 2 else 0.0)
            )
            in_ch = out_ch
        
        self.final = nn.Sequential(
            ConvBlock(in_ch, in_ch, use_bn=use_bn),
            nn.Conv2d(in_ch, n_channels, kernel_size=1)
        )
        
        # Count parameters
        # model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        # self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        # print(f'\nSparseCNNReconstructor has {self.num_params} parameters\n')
    
    def forward(self, G_d: torch.Tensor) -> torch.Tensor:
        """
        G_d: [B, N_obs, 2+C] sparse observations with coordinates and values
        output: [B, C, Num_y, Num_x] reconstructed full field
        """
        B, N_obs, _ = G_d.shape
        G_d_xyu = G_d[..., :2+self.n_channels]
        x = G_d_xyu.reshape(B, -1)  # [B, N_obs * (2+C)]
        
        # MLP embedding to grid
        x = self.mlp_embedding(x)  # [B, 1, H_emb, W_emb]
        
        # Channel embedding
        x = self.channel_embedding(x)  # [B, conv_channels[0], H_emb, W_emb]
        
        # Progressive upsampling through decoder blocks
        for decoder in self.decoders:
            x = decoder(x)
        
        # Resize to exact output size if needed
        if x.shape[2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
        
        # Final convolution
        output = self.final(x)  # [B, C, Num_y, Num_x]
        
        return output

class CNNSparseReconstructor(nn.Module):
    """
    Wrapper module for VAE_Wrapper stage 1 integration.
    Handles multiple time steps and coordinate extraction.
    """
    def __init__(self, cfg: dict, n_channels: int):
        super().__init__()
        
        # Extract configuration
        self.n_channels = n_channels
        self.Num_x = cfg.get('Num_x', 88)
        self.Num_y = cfg.get('Num_y', 300)
        self.N_obs = cfg.get('num_space_sample', 64)
        
        # Create core reconstructor
        self.reconstructor = SparseCNNReconstructor(
            n_channels=self.n_channels,
            N_obs=self.N_obs,
            output_size=(self.Num_y, self.Num_x),
            embedding_size=cfg.get('embedding_size', (8, 8)),
            hidden_dims=cfg.get('hidden_dims', [128, 256]),
            conv_channels=cfg.get('conv_channels', [32, 64, 64, 64, 64]),
            use_bn=False,
            dropout=0.0
        )
    
    def forward(self, G_d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            G_d: [B, T, N_obs, 2+C+...] sparse observations with coordinates
        Returns:
            output: [B, T, C, Num_y, Num_x] reconstructed fields
        """
        B, T, N_obs, _ = G_d.shape
        
        # Process each time step
        outputs = []
        for t in range(T):
            # Extract single time step
            G_d_t = G_d[:, t]  # [B, N_obs, 2+C]
            
            # Reconstruct
            output_t = self.reconstructor(G_d_t)  # [B, C, Num_y, Num_x]
            outputs.append(output_t)
        
        # Stack time steps
        output = torch.stack(outputs, dim=1)      # [B, T, C, Num_y, Num_x]
        
        return output

def unflatten_field_to_image(field_flat: torch.Tensor, Num_y: int, Num_x: int) -> torch.Tensor:
    """
    field_flat: [B, N_pt, C] with N_pt = Num_x * Num_y
    return: [B, C, Num_y, Num_x]
    Assumes row-major flattening: idx = y*Num_x + x
    """
    B, N_pt, C = field_flat.shape
    assert N_pt == Num_x * Num_y, "N_pt must equal Num_x * Num_y"
    img = field_flat.view(B, Num_x, Num_y, C).permute(0, 3, 1, 2).contiguous() # [B, C, Num_x, Num_y]
    return img

def flatten_image_to_field(img: torch.Tensor) -> torch.Tensor:
    """
    img: [B, C, Num_y, Num_x]
    return: [B, N_pt, C] with N_pt = Num_x * Num_y, row-major flattening
    """
    B, C, Num_y, Num_x = img.shape
    field_flat = img.permute(0, 2, 3, 1).contiguous().view(B, Num_y * Num_x, C)
    return field_flat

class VAE_Wrapper(nn.Module):
    """
    Wrapper integrating:
      - field_enc: VAE encoder (image-based) [B, C, H, W] -> mean, logvar
      - field_dec: VAE decoder (image-based) z -> [B, C, H, W]
      - sensor2rep: sparse-to-latent network (G_d, Y) -> mean, logvar
      - decoder_lat: temporal latent propagator

    Stages:
      0) Train VAE on full fields G_f (reconstruction). Handles T_fore >= 1.
      1) Train sparse recon model G_d -> latent -> frozen decoder -> G_f. Handles T_fore >= 1.
      2) Freeze VAE. Encode full fields to latent traj, roll out with temporal model for T_fore > 1, decode to fields.
      x) Downstream: Given sparse obs window (N_window) and T_fore, produce G_f forecast [B, T_fore, N_pt, C].

    Inputs:
      - G_d: [B, T_sample, N_obs, C] sparse observations (N_obs can be <= N_pt)
      - G_f: [B, T_fore, N_pt, C] full fields (may be None in stage x)
      - Y:   [B, N_obs, coord_dim] sparse coordinates
      - U:   [B, N_pt, coord_dim]   full grid coordinates
      - Num_x, Num_y: ints such that N_pt = Num_x * Num_y
      - teacher_force_p: float in [0,1] for temporal model (used in stage 2/x)

    Returns:
      out:  [B, T_fore, N_pt, C] reconstructed/forecast fields
      traj: [B, T_traj, latent_dim] predicted latent trajectory (T_traj depends on stage)
      obs:  [B, T_obs, latent_dim] observed latent trajectory
      mean, logvar: last-step mean/logvar (for loss bookkeeping; aggregate KL outside if needed)
    """
    def __init__(
        self,
        cfg: dict,

        field_enc: nn.Module,
        field_dec: nn.Module,
        sensor2rep: nn.Module,
        decoder_lat: nn.Module,

        latent_dim: int,
        stage: str = "0",
        delta_t: float = 1.0,
        N_window: int = 1,

        use_gaussian_eps: bool = True,
        freeze_vae_in_stage2: bool = True
    ):
        super().__init__()
        self.field_enc = field_enc
        self.field_dec = field_dec
        self.sensor2rep = sensor2rep
        self.decoder_lat = decoder_lat

        self.latent_dim = latent_dim
        self.stage = str(stage)  # "0", "1", "2", or "x"
        self.delta_t = delta_t
        self.N_window = N_window
        self.Num_x = cfg.get('Num_x', 88)
        self.Num_y = cfg.get('Num_y', 300)

        self.use_gaussian_eps = use_gaussian_eps
        self.freeze_vae_in_stage2 = freeze_vae_in_stage2

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor, gaussian: bool = True) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) if gaussian else torch.rand_like(std)
        return mean + eps * std

    def _encode_full_fields(
        self, Gf_flat: torch.Tensor, Num_x: int, Num_y: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode full fields per time step.
        Gf_flat: [B, T, N_pt, C]
        Returns (mean, logvar): each [B, T, latent_dim]
        """
        B, T, N_pt, C = Gf_flat.shape
        means, logvars = [], []
        for t in range(T):
            img = unflatten_field_to_image(Gf_flat[:, t], Num_y, Num_x)  # [B, C, H, W]
            mean_t, logvar_t = self.field_enc(img)  # [B, latent_dim]
            means.append(mean_t)
            logvars.append(logvar_t)
        mean = torch.stack(means, dim=1)
        logvar = torch.stack(logvars, dim=1)
        return mean, logvar

    def _decode_latents_to_fields(
        self, z_seq: torch.Tensor, Num_x: int, Num_y: int
    ) -> torch.Tensor:
        """
        Decode a latent sequence into full fields per time step.
        z_seq: [B, T, latent_dim]
        Returns fields_flat: [B, T, N_pt, C]
        """
        B, T, D = z_seq.shape
        outs = []
        for t in range(T):
            img_t = self.field_dec(z_seq[:, t])  # [B, C, H, W] with H=Num_y, W=Num_x
            field_flat_t = flatten_image_to_field(img_t)  # [B, N_pt, C]
            outs.append(field_flat_t)
        out = torch.stack(outs, dim=1)  # [B, T, N_pt, C]
        return out

    def _latent_observations_from_sparse(self, G_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each time step in G_d, produce mean/logvar and sample z.
        G_d: [B, T_sample, N_obs, C]
        Returns:
          obs_z:   [B, T_sample, latent_dim]
          obs_mu:  [B, T_sample, latent_dim]
          obs_log: [B, T_sample, latent_dim]
        """
        B, T_sample, N_obs, C = G_d.shape
        z_list, mu_list, log_list = [], [], []
        for t in range(T_sample):
            mu_t, log_t = self.sensor2rep(G_d[:, t])  # [B, D]
            z_t = self._reparameterize(mu_t, log_t, gaussian=self.use_gaussian_eps)
            z_list.append(z_t)
            mu_list.append(mu_t)
            log_list.append(log_t)
        obs_z = torch.stack(z_list, dim=1)
        obs_mu = torch.stack(mu_list, dim=1)
        obs_log = torch.stack(log_list, dim=1)
        return obs_z, obs_mu, obs_log

    def forward(
        self,
        G_d: Optional[torch.Tensor],
        G_f: Optional[torch.Tensor],
        Y: Optional[torch.Tensor],
        U: Optional[torch.Tensor],
        teacher_force_p: float = 0.0,
        T_fore_override: Optional[int] = None,
    ):
        """
        Stage-specific behavior:

        stage "0": Train VAE on G_f
          - Input:  G_f [B, T_fore, N_pt, C]
          - Output: out [B, T_fore, N_pt, C]; traj=None; obs=None

        stage "1": Train sparse reconstruction G_d -> latent -> decoder -> G_f
          - Input:  G_d [B, T_sample, N_obs, C], Y, and target G_f for loss
          - Output: out [B, T_fore, N_pt, C]; traj=None; obs=None

        stage "2": Freeze VAE; train temporal latent propagator
          - Input:  G_f [B, T_total, N_pt, C] where T_total >= N_window, and T_fore>1
          - Output: out [B, T_fore, N_pt, C]; traj [B, T_total_latent, D]; obs [B, N_window, D]

        stage "x": Downstream forecasting from sparse obs window to full spatiotemporal distributions
          - Input:  G_d [B, N_window, N_obs, C], Y, T_fore (via T_fore_override)
          - Output: out [B, T_fore, N_pt, C]; traj; obs
        """
        stage = self.stage
        last_mean = None
        last_logvar = None

        if stage == "0":
            # VAE training on full fields
            assert G_f is not None, "Stage 0 requires G_f"
            B, T_fore, N_pt, C = G_f.shape
            # Encode each time step
            mean_seq, logvar_seq = self._encode_full_fields(G_f, self.Num_x, self.Num_y)  # [B, T, D]
            # Sample per time step
            z_seq = self._reparameterize(mean_seq, logvar_seq, gaussian=self.use_gaussian_eps)
            # Decode back to fields
            out = self._decode_latents_to_fields(z_seq, self.Num_x, self.Num_y)  # [B, T, N_pt, C]
            # print(f'out.shape is {out.shape}')
            last_mean = mean_seq[:, -1]
            last_logvar = logvar_seq[:, -1]
            traj = z_seq
            obs = None
            return out, traj, obs, last_mean, last_logvar

        elif stage == "1":

            out = self.sensor2rep(G_d)  # [B, T, N_pt, C]

            B, T, C, Ny, Nx = out.shape
            out = out.permute(0, 1, 3, 4, 2).contiguous()  # [B, T, Num_y, Num_x, C]
            out = out.view(B, T, Ny * Nx, C)               # [B, T, N_pt, C]

            last_mean = None
            last_logvar = None
            traj = None
            obs = None
            return out, traj, obs, last_mean, last_logvar

        elif stage == "2":
            # Train temporal propagator with frozen VAE
            assert G_f is not None, "Stage 2 requires full fields G_f"
            B, T_total, N_pt, C = G_f.shape
            assert self.N_window <= T_total, "N_window must be <= total time length"
            # Freeze (no grad) encoder/decoder if requested
            if self.freeze_vae_in_stage2:
                self.field_enc.eval()
                self.field_dec.eval()
                for p in self.field_enc.parameters():
                    p.requires_grad_(False)
                for p in self.field_dec.parameters():
                    p.requires_grad_(False)

            # Encode all timesteps to latent
            mean_seq, logvar_seq = self._encode_full_fields(G_f, self.Num_x, self.Num_y)  # [B, T_total, D]
            z_seq = self._reparameterize(mean_seq, logvar_seq, gaussian=self.use_gaussian_eps)  # [B, T_total, D]

            # Observed window
            obs = z_seq[:, :self.N_window, :]  # [B, N_window, D]
            # Determine forecast horizon
            if T_fore_override is not None:
                T_fore = int(T_fore_override)
            else:
                # If G_f includes target segment, we can set T_fore = T_total - N_window
                T_fore = max(1, T_total - self.N_window)

            # Rollout with temporal model
            # Expected interface: decoder_lat(obs, T_fore, teacher_force_p, maybe targets)
            # If teacher forcing requires targets, we pass the next latents from z_seq as teacher (when available).
            future_target = None
            if z_seq.shape[1] >= self.N_window + T_fore:
                future_target = z_seq[:, self.N_window:self.N_window + T_fore, :]  # [B, T_fore, D]

            traj_pred = self.decoder_lat(
                obs=obs,  # [B, N_window, D]
                T_fore=T_fore,
                teacher_force_p=teacher_force_p,
                target_future=future_target,  # Optional, depends on your implementation
                delta_t=self.delta_t,
            )  # Expected [B, N_window + T_fore, D] or [B, T_fore, D]

            # Normalize output shape: make traj contain both context and forecast
            if traj_pred.shape[1] == T_fore:
                traj = torch.cat([obs, traj_pred], dim=1)  # [B, N_window+T_fore, D]
                forecast_lat = traj_pred  # [B, T_fore, D]
            else:
                traj = traj_pred  # [B, N_window+T_fore, D]
                forecast_lat = traj[:, -T_fore:, :]

            # Decode forecast latents to fields
            out = self._decode_latents_to_fields(forecast_lat, self.Num_x, self.Num_y)  # [B, T_fore, N_pt, C]

            last_mean = mean_seq[:, -1]
            last_logvar = logvar_seq[:, -1]
            return out, traj, obs, last_mean, last_logvar

        elif stage == "x":
            # Stage x: Downstream pipeline: sparse obs window -> latent rollout -> decode fields

            B, T_obs, N_obs, C = G_d.shape
            assert T_obs == self.N_window, "G_d time length must equal N_window in stage x or set N_window accordingly"
            # Freeze VAE
            self.field_enc.eval()
            self.field_dec.eval()
            for p in self.field_enc.parameters():
                p.requires_grad_(False)
            for p in self.field_dec.parameters():
                p.requires_grad_(False)

            # Encode sparse obs window to latent
            obs_z, obs_mu, obs_log = self._latent_observations_from_sparse(G_d)  # [B, N_window, D]
            obs = obs_z

            # Forecast horizon must be provided
            assert T_fore_override is not None, "Provide T_fore_override for stage x"
            T_fore = int(T_fore_override)

            traj_pred = self.decoder_lat(
                obs=obs, T_fore=T_fore, teacher_force_p=teacher_force_p, target_future=None, delta_t=self.delta_t
            )  # [B, N_window+T_fore, D] or [B, T_fore, D]

            if traj_pred.shape[1] == T_fore:
                traj = torch.cat([obs, traj_pred], dim=1)
                forecast_lat = traj_pred
            else:
                traj = traj_pred
                forecast_lat = traj[:, -T_fore:, :]

            out = self._decode_latents_to_fields(forecast_lat, self.Num_x, self.Num_y)  # [B, T_fore, N_pt, C]

            last_mean = obs_mu[:, -1, :]
            last_logvar = obs_log[:, -1, :]
            return out, traj, obs, last_mean, last_logvar

        else:
            raise ValueError(f"Invalid stage: {stage}")
        
# ________________________________________________________

