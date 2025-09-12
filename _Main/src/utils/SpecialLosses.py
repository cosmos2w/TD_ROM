
import torch
import torch.fft
import torch.nn.functional as F
from typing import Callable, Optional, Tuple
# from pytorch_wavelets import DWTForward

# ---------- helper: scatter point cloud to regular grid ----
def cloud_to_grid(values: torch.Tensor,
                  coords: torch.Tensor,
                  grid_size: tuple[int, int]) -> torch.Tensor:
    """
    values : [B,T,P,C]
    coords : [B,T,P,2]  (normalised to [0,1]×[0,1] or broadcast there)
    return : [B,T,C,H,W]
    """
    B, T, P, C = values.shape
    H, W       = grid_size
    device     = values.device
    dtype      = values.dtype

    # ── 1. make coords time-compatible
    if coords.ndim == 3:
        coords = coords[:, None].expand(-1, T, -1, -1)   # [B,T,P,2]

    # ── 2. integer pixel indices
    y = (coords[..., 1] * (H - 1)).round().clamp(0, H - 1).long()  # row
    x = (coords[..., 0] * (W - 1)).round().clamp(0, W - 1).long()  # col

    # ── 3. flatten everything except channel
    b = torch.arange(B, device=device)[:, None, None].expand(B, T, P).reshape(-1)
    t = torch.arange(T, device=device)[None, :, None].expand(B, T, P).reshape(-1)
    y = y.reshape(-1)
    x = x.reshape(-1)

    # repeat each (b,t,y,x) tuple for every channel
    c = torch.arange(C, device=device)
    b = b.repeat_interleave(C)
    t = t.repeat_interleave(C)
    y = y.repeat_interleave(C)
    x = x.repeat_interleave(C)
    c = c.repeat(B * T * P)

    # ── 4. gather the values to scatter
    vals = values.permute(0, 1, 2, 3).reshape(-1)   # same order as (b,t,c,y,x)

    # ── 5. scatter-add into the grid
    grid = torch.zeros(B, T, C, H, W, dtype=dtype, device=device)
    grid.index_put_((b, t, c, y, x), vals, accumulate=True)
    return grid

def select_high_freq_sensors(
    G: torch.Tensor,
    k: int,
    field_chan: int = 2,
    method: str = "mad",
    hp_kernel: torch.Tensor = None
):
    """
    Select the top-k "most variable" sensors in G according to `method`.

    Args:
      G           (B, T, P, C): input batch
      k           int: how many sensors to keep
      field_chan  int: which channel holds the field value (default=2)
      method      str: one of {"mad","std","hp","ptp"}
      hp_kernel   (optional) torch.Tensor [1,1,L]: kernel for HP filter

    Returns:
      G_sel  (B, T, k, C): G restricted to top-k sensors
      idx    (k,)        : indices of selected sensors
      score  (P,)        : variability score for each of the P sensors
    """
    B, T, P, C = G.shape
    V = G[..., field_chan]          # [B, T, P]

    if method.lower() == "mad":
        # mean absolute difference
        Delta = (V[:,1:,:] - V[:,:-1,:]).abs()  # [B, T-1, P]
        var_t = Delta.mean(dim=1)               # [B, P]
        score = var_t.mean(dim=0)           # [P]

    elif method.lower() == "std":
        # standard deviation over time
        var_t = V.std(dim=1, unbiased=False)  # [B, P]
        score = var_t.mean(dim=0)             # [P]
        # print(f'score is {score}')

    elif method.lower() == "ptp":
        # peak-to-peak range
        max_t = V.max(dim=1).values           # [B, P]
        min_t = V.min(dim=1).values           # [B, P]
        ptp  = max_t - min_t                  # [B, P]
        score = ptp.mean(dim=0)               # [P]

    elif method.lower() == "hp":
        # high-pass filter energy
        # default second-derivative kernel [-1,2,-1]
        if hp_kernel is None:
            hp_kernel = torch.tensor([[-1., 2., -1.]],
                                     device=G.device,
                                     dtype=G.dtype).view(1,1,3)
        # reshape to (B*P, 1, T)
        Vp = V.permute(0,2,1).reshape(B*P, 1, T)
        # conv1d with padding 'same'
        pad = (hp_kernel.shape[2] // 2,)
        hp = F.conv1d(Vp, hp_kernel, padding=pad)   # [B*P,1,T]
        energy = hp.abs().mean(dim=2).reshape(B, P) # [B, P]
        score = energy.mean(dim=0)                  # [P]

    else:
        raise ValueError(f"Unknown method: {method!r}, choose from mad,std,hp,ptp")

    # pick top‐k
    # k = min(k, P)
    # _, idx = torch.topk(score, k=k, dim=0)  # [k]
    _, idx_all = torch.sort(score, descending=False)  # now lowest scores (ties) → first
    idx = idx_all[-k:]                                # take the k *highest* scores
    G_sel = G[:, :, idx, :]

    # print(f'idx is {idx}')

    # gather them out
    G_sel = G[:,:,idx,:]                    # [B, T, k, C]
    return G_sel, idx, score

# ---------- Sobolev-H^s loss --------------------------------

def _grad_field(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Returns Grad_x f;  f: [B,T,P,C]  -->  [B,T,P,C,2]"""
    grads = []
    for d in range(2):
        g, = torch.autograd.grad(
                outputs=f,
                inputs=x,
                grad_outputs=torch.ones_like(f),
                retain_graph=True,
                create_graph=True,
                allow_unused=False)[0][..., d].chunk(1, dim=-1)
        grads.append(g)
    return torch.stack(grads, dim=-1)  # last dim is x/y

def _laplacian_field(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Laplacian of f via trace  -> [B,T,P,C]"""
    lap = 0.
    for d in range(2):
        g, = torch.autograd.grad(
                outputs=f,
                inputs=x,
                grad_outputs=torch.ones_like(f),
                retain_graph=True,
                create_graph=True,
                allow_unused=False)[0][..., d].chunk(1, dim=-1)
        # second derivative wrt the same coordinate
        h, = torch.autograd.grad(
                outputs=g,
                inputs=x,
                grad_outputs=torch.ones_like(g),
                retain_graph=True,
                create_graph=True,
                allow_unused=False)
        lap = lap + h[..., d]
    return lap

class SobolevHsLoss(torch.nn.Module):
    """
    Sobolev loss up to derivative order s (currently s ∈ {0,1,2}).
    L = \Sum_{|alpha|≤s} lambda_{|alpha|} ∫ w(x) |∂^alpha f~ - ∂^alpha f|^2 dx
    Need coords & quadrature weights when data are not on a tensor product grid.
    """
    def __init__(self,
                 s: int = 1,
                 lambda_der: Optional[Tuple[float, ...]] = None):
        super().__init__()
        assert s in (0,1,2), "Only s=0,1,2 implemented."
        self.s = s
        if lambda_der is None:
            lambda_der = tuple(1.0 for _ in range(s+1))
        assert len(lambda_der) == s+1
        self.register_buffer('lambda_der',
                             torch.tensor(lambda_der, dtype=torch.float32))

    def forward(self,
                pred:   torch.Tensor,        # [B,T,P,C]
                target: torch.Tensor,        # [B,T,P,C]
                coords: Optional[torch.Tensor] = None,  # [B,T,P,2]
                weight: Optional[torch.Tensor] = None   # [B,T,P]
               ) -> torch.Tensor:

        if weight is None:
            weight = torch.ones_like(pred[...,:1])  # [B,T,P]

        # order-0 term (plain weighted MSE)
        print(f'pred.shape is {pred.shape}')
        print(f'target.shape is {target.shape}')
        mse0 = torch.mean(weight * (pred - target).pow(2))

        if self.s == 0:
            return self.lambda_der[0] * mse0

        # Need gradients w.r.t coordinates
        assert coords is not None, "coords required for derivative terms"
        if not coords.requires_grad:
            coords = coords.clone().requires_grad_(True)

        # First-order derivatives
        grad_pred   = _grad_field(pred, coords)   # [B,T,P,C,2]
        grad_target = _grad_field(target, coords)

        mse1 = torch.mean(weight[..., None, None] * (grad_pred - grad_target).pow(2))

        if self.s == 1:
            return self.lambda_der[0]*mse0 + self.lambda_der[1]*mse1

        # Second-order (Laplacian) – cheap but crude
        lap_pred   = _laplacian_field(pred, coords)
        lap_target = _laplacian_field(target, coords)
        mse2 = torch.mean(weight * (lap_pred - lap_target).pow(2))

        return (self.lambda_der[0]*mse0 +
                self.lambda_der[1]*mse1 +
                self.lambda_der[2]*mse2)

# ---------- Wavelet multiresolution loss -------------------

class WaveletMSSLoss(torch.nn.Module):
    """
    Multiresolution L2 between wavelet coefficients.
    Works on regular grids only.
    Hyper-parameters
        wave         : name of wavelet (e.g. 'db4', 'haar', 'sym6')
        J            : number of decomposition levels
        scale_weights: length-(J+1) tuple (low-pass + J high-pass levels)
    """
    def __init__(self,
                 wave: str = 'db4',
                 J: int = 3,
                 scale_weights: Optional[Tuple[float, ...]] = None):
        super().__init__()
        assert DWTForward is not None, "Install pytorch_wavelets"
        self.dwt = DWTForward(J=J, wave=wave, mode='zero')
        self.J = J
        if scale_weights is None:
            scale_weights = (1.0,) + tuple(1.0 for _ in range(J))
        assert len(scale_weights) == J+1
        self.register_buffer('ws', torch.tensor(scale_weights))

    def forward(self,
                pred:   torch.Tensor,           # [B,T,P,C]  (P=H*W)
                target: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                grid_size: Tuple[int,int] = (128,128)) -> torch.Tensor:

        if pred.ndim == 4:   # scatter point clouds
            assert coords is not None, "coords needed for point clouds"
            pred   = cloud_to_grid(pred,   coords, grid_size)  # -> [B,T,C,H,W]
            target = cloud_to_grid(target, coords, grid_size)

        B,T,C,H,W = pred.shape
        pred = pred.reshape(B*T, C, H, W)
        target = target.reshape(B*T, C, H, W)

        yl_p, yh_p = self.dwt(pred)
        yl_t, yh_t = self.dwt(target)

        loss = self.ws[0] * F.mse_loss(yl_p, yl_t)
        for j in range(self.J):
            loss = loss + self.ws[j+1] * F.mse_loss(yh_p[j], yh_t[j])
        return loss

# ---------- Weighted Fourier MSE loss ----------------------
class WeightedFourierMSE(torch.nn.Module):
    """
    L = \Sum_k  w(k) | F~(k) - F(k) |^2
    Hyper-parameters
        weight_fn(kx,ky) -> tensor of weights, or one of
            'uniform'        : w=1
            'sobolev_s'      : w = (1+|k|^2)^{s}
            'power_p'        : w = |k|^{p}
    """
    def __init__(self,
                 weighting: str | Callable = 'uniform',
                 s_or_p: float = 1.0):
        super().__init__()
        self.weighting = weighting
        self.s_or_p = s_or_p

    def _weights(self, H: int, W: int, device) -> torch.Tensor:

        ky = torch.fft.fftfreq(H, d=1./H, device=device).view(-1,1)
        kx = torch.fft.fftfreq(W, d=1./W, device=device).view(1,-1)
        k_norm2 = kx**2 + ky**2
        
        if callable(self.weighting):
            return self.weighting(kx,ky)
        if self.weighting == 'uniform':
            return torch.ones_like(k_norm2)
        if self.weighting == 'sobolev_s':
            return (1. + k_norm2) ** (self.s_or_p)
        if self.weighting == 'power_p':
            return (k_norm2 + 1e-9) ** (0.5*self.s_or_p)
        raise ValueError("Unknown weighting")

    def forward(self,
                pred:   torch.Tensor,           # [B,T,P,C] or [B,T,C,H,W]
                target: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                grid_size: Tuple[int,int] = (300,88)) -> torch.Tensor:

        if pred.ndim == 4:   # point cloud -> grid
            assert coords is not None
            pred   = cloud_to_grid(pred,   coords, grid_size)
            target = cloud_to_grid(target, coords, grid_size)

        B,T,C,H,W = pred.shape
        pred   = pred.reshape(B*T, C, H, W)
        target = target.reshape(B*T, C, H, W)

        # Forward FFT  (orthonormal normalisation)
        Fp = torch.fft.rfftn(pred, dim=(-2,-1), norm='ortho')
        Ft = torch.fft.rfftn(target, dim=(-2,-1), norm='ortho')

        # Build weight matrix once per call
        Wfull = self._weights(H, W, pred.device)      # (H, W)
        Whalf = Wfull[:, :W//2 + 1]                   # rfft output half-plane

        loss = torch.mean(Whalf * (Fp - Ft).abs()**2)
        return loss

# ---------- factory wrapper --------------------------------
def get_loss(name: str, **kwargs):
    """
    Special loss constructor, name: {'sobolev','wavelet','fourier'}
    """
    name = name.lower()
    if name == 'sobolev':
        return SobolevHsLoss(**kwargs)
    if name == 'wavelet':
        return WaveletMSSLoss(**kwargs)
    if name == 'fourier':
        return WeightedFourierMSE(**kwargs)
    raise ValueError(f'Unknown loss {name}')

# REVISED 0818: Added assistant functions for PSD and spectrum (requirement 1)
def compute_psd(field, Y, grid_size: int = 128, n_bins: int = 64, eps: float = 1e-8):
    """
    field : [B, 1, N_pt, N_c]   values at scattered points
    Y     : [N_pt, 2]           (x,y) coords in the range [-1, 1]

    Returns
    -------
    psd   : [B, n_bins]         isotropic, channel-averaged PSD (normalized)
    """
    B, _, N_pt, N_c = field.shape               # batch, points, channels
    device = field.device

    # ---------------------------------------------------------------
    # 1.  Rasterise scattered points → regular grid (differentiable)
    # ---------------------------------------------------------------
    # Map coords in [-1,1] to integer pixel indices in [0, grid_size-1]
    xi = ((Y[:, 0] + 1) * 0.5 * (grid_size - 1)).long()     # [N_pt]
    yi = ((Y[:, 1] + 1) * 0.5 * (grid_size - 1)).long()     # [N_pt]
    flat_idx = yi * grid_size + xi                          # [N_pt]  (row-major)

    # Per-cell counter so we can average when several points fall in the same pixel
    counts = torch.zeros(grid_size * grid_size,
                         device=device).scatter_add_(0, flat_idx,
                                                      torch.ones_like(flat_idx, dtype=torch.float))

    # Avoid division by zero; keep counts ≥ 1
    counts = counts.clamp_min(1.0)

    # Build the grid tensor: [B, N_c, H*W]
    grid_field = torch.zeros(B, N_c, grid_size * grid_size, device=device)
    for b in range(B):
        # field[b, 0] : [N_pt, N_c]   →  transpose to [N_c, N_pt] for scatter_add_
        grid_field[b] = grid_field[b].scatter_add_(1,
                                                   flat_idx.expand(N_c, N_pt),
                                                   field[b, 0].transpose(0, 1))
    # Average duplicates
    grid_field = grid_field / counts            # broadcasting over [H*W]

    # Reshape to images  [B, N_c, H, W]
    grid_field = grid_field.view(B, N_c, grid_size, grid_size)

    # REVISED: Optional smoothing for empty cells (Gaussian blur to reduce artifacts; comment out if not needed)
    # kernel = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float, device=device).view(1,1,3,3) / 16
    # grid_field = F.conv2d(grid_field.permute(0,1,2,3), kernel, padding=1).permute(0,2,3,1)  # Assumes N_c=1; adjust for multi-channel

    # ---------------------------------------------------------------
    # 2.  2-D FFT → power spectrum (channel-average, NORMALIZED)
    # ---------------------------------------------------------------
    fft2 = torch.fft.fftn(grid_field, dim=(-2, -1))         # [B, N_c, H, W]
    power = fft2.abs() ** 2
    power = power.mean(dim=1)                               # [B, H, W]
    power = power / (grid_size ** 2 + eps)                  # REVISED: Normalize by grid points for density

    # ---------------------------------------------------------------
    # 3.  Isotropic binning → 1-D PSD
    # ---------------------------------------------------------------
    ky, kx = torch.meshgrid(torch.fft.fftfreq(grid_size, d=1.0),
                            torch.fft.fftfreq(grid_size, d=1.0),
                            indexing='ij')
    k_mag = torch.sqrt(kx ** 2 + ky ** 2).to(device).flatten()      # [H*W]

    power_flat = power.view(B, -1)                                  # [B, H*W]

    k_max = k_mag.max()
    bins = torch.linspace(0, k_max, n_bins + 1, device=device)      # n_bins+1 edges
    psd = torch.zeros(B, n_bins, device=device)

    for i in range(n_bins):
        mask = (k_mag >= bins[i]) & (k_mag < bins[i + 1])           # [H*W] bool
        if mask.any():
            psd[:, i] = power_flat[:, mask].mean(dim=1)
        # else leave as zero

    return psd.clamp(min=eps)  # REVISED: Clamp for stability in log/loss


def compute_spectrum(time_series, subsample_pts=512, n_bins=None, eps: float = 1e-8):
    """time_series: [B, T, N_pt, N_c]. Subsamples N_pt to subsample_pts, computes aggregated spectrum."""
    B, T, N_pt, N_c = time_series.shape
    device = time_series.device
    # Subsample points for efficiency (deterministic for reproducibility; use fixed indices)
    subsample_idx = torch.linspace(0, N_pt-1, subsample_pts, dtype=torch.long, device=device)
    ts = time_series[:, :, subsample_idx, :]  # [B, T, subsample_pts, N_c]
    
    # REVISED: Normalize input ts per batch/channel to zero-mean unit-variance (optional; helps scale)
    # ts = (ts - ts.mean(dim=1, keepdim=True)) / (ts.std(dim=1, keepdim=True) + eps)
    
    fft = torch.fft.fft(ts, dim=1)  # [B, T, subsample_pts, N_c]
    power = torch.abs(fft[:, :T//2+1])**2  # [B, T//2+1, subsample_pts, N_c]
    power = power.mean(dim=(2,3)) / (T + eps)  # [B, T//2+1] (aggregated and normalized by T)
    
    if n_bins:  # REVISED: Mask-based binning like PSD for better frequency distribution
        freq = torch.fft.fftfreq(T, d=1.0)[:T//2+1].to(device)  # [T//2+1] frequencies
        bins = torch.linspace(0, 0.5, n_bins + 1, device=device)
        power_binned = torch.zeros(B, n_bins, device=device)
        for i in range(n_bins):
            mask = (freq >= bins[i]) & (freq < bins[i+1])
            if mask.any():
                power_binned[:, i] = power[:, mask].mean(dim=1)
        return power_binned.clamp(min=eps)  # [B, n_bins]
    return power.clamp(min=eps)  # [B, T//2+1]