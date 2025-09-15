
"""
Utilities that turn the *repository-standard* HDF5 file

    data_dict = {
        "fields"    : [B, N_t, N_x, N_y, N_z, F]   (u-values)       
        "coordinates": [N_x, N_y, N_z, N_dim]      (x,y, for 2D cases z=0)
        "time"      : [N_t]
    }
    
into the three tensors expected by TD-ROM:

    G_down   : [B, N_ts, N_xs, 4]    - (x,y,u,t)      <- encoder input
    G_down_t : [B, N_ts, N_pts, 1]   - (· · ·)        <- DS only
    G_full   : [B, N_full, N_pts, 1] - (· · ·)        <- training target
    U        : [B, N_para]                            <- Conditional parameters

    If no U vector in the dataset, use a dummy tensor.
"""

from __future__ import annotations
import pathlib, random, h5py, torch, datetime, numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] 
DATA_ROOT = REPO_ROOT 

def data_path(*parts) -> pathlib.Path:
    return DATA_ROOT.joinpath(*parts)

def _load_h5_to_ram(fp: pathlib.Path, 
                    channel: int,
                    process_mode: str = "None",):

    fp = pathlib.Path(fp)
    if not fp.is_absolute():
        fp = (REPO_ROOT / fp).resolve()  # or use DATA_ROOT if fp is relative to repo root
    if not fp.exists():
        raise FileNotFoundError(f"Data file not found: {fp}")

    with h5py.File(fp, "r") as f:

        fields_np  = f["fields"][...].astype("float32")       
        coords  = f["coordinates"][...].astype("float32")   
        t_vec   = f["time"][...].astype("float32")      

        if "conditions" in f.keys():
            U   = f["conditions"][...].astype("float32")
        else:
            U   = np.zeros((fields_np.shape[0], 1), np.float32)

        # optional original mean/std
        mean_np = f["orig_mean"][...].astype("float32") if "orig_mean" in f else None
        std_np  = f["orig_std"][...].astype("float32")  if "orig_std"  in f else None
    # UN-STANDARDIZE (if available)
    if mean_np is not None and std_np is not None:
        # mean_np (1, N_x, N_y, F) / std_np (1, 1, 1, F) / fields_np (1, N_t, N_x, N_y, N_z, F)
        Nx, Ny, F = mean_np.shape[1], mean_np.shape[2], mean_np.shape[3]
        mean_np = mean_np.reshape((1, 1, Nx, Ny, 1, F))
        fields_np = fields_np * std_np + mean_np

    # reshape, normalise, flatten once -----------------------------------
    fields = torch.from_numpy(fields_np).squeeze(4)  # (B,N_t,N_x,N_y,2)
    print(f'fields.shape is {fields.shape}')

    B, N_t, N_x, N_y, N_c = fields.shape
    N_pts = N_x * N_y

    # select channel(s)
    if channel == -1:
        # keep all channels → (B,N_t,N_pts,N_c)
        u_vals = fields.view(B, N_t, N_pts, N_c)
    else:
        # single channel → (B,N_t,N_pts,1)
        u_vals = fields[..., channel].view(B, N_t, N_pts).unsqueeze(-1)

    dims = (0,1,2)
    # apply chosen preprocessing
    if process_mode == "MinMaxNorm":

        print('Using Min-Max normalization to all selected channels')
        # reduce over batch, time and spatial dims
        mins = u_vals.amin(dim=(0,1,2), keepdim=True)  # -> (1,1,1,N_c)
        maxs = u_vals.amax(dim=(0,1,2), keepdim=True)  # -> (1,1,1,N_c)
        print(f"mins: {mins.squeeze()}   maxs: {maxs.squeeze()}")
        u_vals = (u_vals - mins) / (maxs - mins + 1e-8)

    elif process_mode == "MeanStdStand":
        print('Using Mean-Std standardization to all selected channels for pre-processing!\n')
        means = u_vals.mean(dim=dims, keepdim=True)
        stds  = u_vals.std(dim=dims, unbiased=False, keepdim=True)
        print(f'means are {means}, stds is {stds}')
        u_vals = (u_vals - means) / (stds + 1e-8)

    elif process_mode == "SymLogQuant":
        print("Symmetric-log with 1/99 quantile scaling (recommended for turbulence)")
        abs_vals = u_vals.abs()                                  # (B,T,N_pts,N_c)
        abs_flat = abs_vals.view(-1, abs_vals.shape[-1])         # (B*T*N_pts, N_c)
        total      = abs_flat.shape[0]
        max_samples = 10_000_000
        if total > max_samples:
            idx   = torch.randint(total, (max_samples,))
            sample = abs_flat[idx]
        else:
            sample = abs_flat
        sample_cpu = sample.cpu()

        q1  = torch.quantile(sample_cpu, 0.02, dim=0)     # (N_c,)
        q99 = torch.quantile(sample_cpu, 0.98, dim=0)     # (N_c,)
        scale = (q99 - q1).clamp_min(1e-8)

        u_scaled = (abs_vals - q1) / scale
        u_scaled = torch.clamp(u_scaled, min=1e-8, max=1.0)
        u_vals = torch.sign(u_vals) * (-torch.log(u_scaled))

    elif process_mode == "None":
        print('Using NO pre-processing to fields data!\n')
        pass
    else:
        raise ValueError(f"Unknown process_mode {process_mode!r}")

    coords = torch.from_numpy(coords).squeeze(2)[..., :2]  # (N_x,N_y,2)
    xy = coords.view(-1, 2)                                # (N_pts,2)
    xy_min, xy_max = xy.min(0).values, xy.max(0).values

    xy = (xy - xy_min) / (xy_max - xy_min)
    xy = xy * 2.0 - 1.0                                    # converted to [-1, 1]

    t_vec = torch.from_numpy(t_vec)
    U_vec = torch.from_numpy(U)

    return u_vals, xy, t_vec, U_vec

class H5SpatialTemporalDatasetConcentrated(Dataset):

    def __init__(self,
                 h5_file: str | pathlib.Path,
                 num_time_sample: int,
                 num_space_sample: int,
                 multi_factor: float = 2,
                 channel: int = 0,
                 process_mode: str = "None",
                 num_samples: int = 1,
                 Full_Field_DownS : float = 0.50,
                 *,
                 split: str = "train",
                 train_ratio: float = 0.9,
                 global_restriction: bool = False,
                 sample_restriction : bool = False,          
                 sample_params      : dict | None = None,    
                 ):
        
        super().__init__()
        self.process_mode = process_mode
        self.u, self.xy, self.t, self.U = _load_h5_to_ram(pathlib.Path(h5_file), channel, self.process_mode)
        self.B, self.N_t, self.N_pts, self.N_c = self.u.shape
        
        self.num_time_sample  = num_time_sample
        self.num_space_sample = num_space_sample
        self.num_samples      = num_samples
        self.T_full           = int(num_time_sample * multi_factor)
        self.split            = split.lower()
        self.idx_t_train      = int(self.N_t * train_ratio) - 1

        self.Cut_Time = 1.0

        self.Full_Field_DownS = Full_Field_DownS
        self.global_restriction = bool(global_restriction)                  

        # pre-compute broadcastable views once ---------------------------
        self.xy_exp = self.xy                           # (N_pts,2)
        self.t_exp  = self.t                            # (N_t,)

        self.sample_restriction = bool(sample_restriction)
        self.sample_params      = sample_params or {}   # empty dict if None
        if self.sample_restriction:
            x_lo = self.sample_params.get("x_lo", 0.0)
            x_hi = self.sample_params.get("x_hi", 1.0)
            y_lo = self.sample_params.get("y_lo", 0.0)
            y_hi = self.sample_params.get("y_hi", 1.0)
        else:                            
            # no spatial masking requested
            x_lo, y_lo = -float("inf"), -float("inf")
            x_hi, y_hi =  float("inf"),  float("inf")

        mask = (
        (self.xy[:,0] >= x_lo) & (self.xy[:,0] <= x_hi) &
        (self.xy[:,1] >= y_lo) & (self.xy[:,1] <= y_hi)
        )
        self.region_idx = torch.nonzero(mask, as_tuple = True)[0]
 
        # --------- pool of indices we are allowed to sample ---------------
        if self.global_restriction:
            print(f'global_restriction is {self.global_restriction}, will restrict reconstruction region globally!\n')
            self.recon_pool = self.region_idx           # restricted pool
        else:
            print(f'global_restriction is {self.global_restriction}, will rebuild the COMPLETE domain!\n')
            self.recon_pool = torch.arange(self.N_pts)  # full pool

        if self.Full_Field_DownS >= 1.0 - 1e-6:
            self.Num_recon_pts = len(self.recon_pool)
        else:
            self.Num_recon_pts = int(
                len(self.recon_pool) * self.Full_Field_DownS)

    def __len__(self): return self.B * self.num_samples

    def _rand_time_start(self):
        if self.split == "train":
            t_min, t_max = 0, int(self.Cut_Time * self.idx_t_train) - self.T_full + 1
        else:
            t_min, t_max = int(self.Cut_Time * self.idx_t_train) + 1, int(self.Cut_Time * self.N_t) - self.T_full
        return random.randint(t_min, t_max)

    def __getitem__(self, idx: int):

        case_idx   = idx // self.num_samples
        sample_idx = idx % self.num_samples
        u_case = self.u[case_idx]          # (N_t, N_pts)
        U_case = self.U[case_idx]          # (N_para,)

        t0 = self._rand_time_start()

        t_full_idx  = slice(t0, t0 + self.T_full)           # length T_full
        t_obs_idx   = slice(t0, t0 + self.num_time_sample)  # length N_ts

        # --------- reconstruction points (used by  G_full and G_down_t) ---
        if self.Full_Field_DownS >= 1.0 - 1e-6:
            recon_idx = self.recon_pool
        else:
            perm       = torch.randperm(len(self.recon_pool),
                                        device=self.recon_pool.device)
            recon_idx  = self.recon_pool[perm[: self.Num_recon_pts]]

        # --------- observation points (used by  G_down) -----------------
        perm      = torch.randperm(len(self.region_idx), device=self.region_idx.device)
        local_idx = perm[: self.num_space_sample]     # positions inside region_idx
        obs_idx   = self.region_idx[local_idx]        # absolute indices in full grid

        recon_idx, _ = recon_idx.sort()
        obs_idx,   _ = obs_idx.sort()

        def gather(u, t_slice, s_idx):
            return u[t_slice][:, s_idx, :]            
        # G_full & G_down_t : can be all channels
        G_full_u = gather(u_case, t_full_idx, recon_idx)
        G_down_t = gather(u_case, t_obs_idx,  recon_idx)
        G_down_u = gather(u_case, t_obs_idx,  obs_idx )

        # coordinates (static) -------------------------------------------
        Y_recon = self.xy_exp[recon_idx]          # (Nr,2)
        Y_obs   = self.xy_exp[obs_idx]            # (Ns,2)

        # time coordinates ------------------------------------------------
        t_full  = self.t_exp[t_full_idx]          # (T_full,)
        t_obs   = self.t_exp[t_obs_idx]           # (N_ts,)

        # Build (2/3+N_c+1)-channel tensors on the *fly* ---------------------
        def build_cube(u_tensor, y, t_vec):
            # u_tensor : (T,Ns,N_para), y : (Ns,2), t_vec : (T,)
            T, Ns = u_tensor.shape[:2]
            y2 = y.unsqueeze(0).expand(T, -1, -1)           # (T,Ns,2)
            t2 = t_vec.view(-1,1,1).expand(-1, Ns, 1)       # (T,Ns,1)
            return torch.cat((y2, u_tensor, t2), dim=-1)    # (T,Ns,2+N_c+1)

        # We include sample position and time information in downsampled tensor G_down
        G_down = build_cube(G_down_u, Y_obs,  t_obs) 
        G_full = G_full_u                

        return (G_down.float(),   
                G_down_t.float(), 
                G_full.float(),   
                Y_recon.float(),  
                U_case.float())   

def make_loaders(
    h5_file,
    *,
    num_time_sample : int,
    num_space_sample: int,

    multi_factor: float = 2,
    train_ratio : float = 0.9,
    batch_size  : int = 64,
    workers: int = 4,        # >0 !
    channel: int = 0,
    process_mode : str = "None",
    num_samples  : int = 1,
    Full_Field_DownS: float = 0.5,

    global_restriction : bool = False,
    sample_restriction : bool = False,   
    sample_params      : dict | None = None, 
):
    
    ds_train = H5SpatialTemporalDatasetConcentrated(
        h5_file, num_time_sample, num_space_sample,
        multi_factor, channel, process_mode, num_samples, Full_Field_DownS, split="train", 
        train_ratio=train_ratio, global_restriction = global_restriction, 
        sample_restriction = sample_restriction, sample_params = sample_params)
    
    ds_test  = H5SpatialTemporalDatasetConcentrated(
        h5_file, num_time_sample, num_space_sample,
        multi_factor, channel, process_mode, int(num_samples/5), Full_Field_DownS, split="test",  
        train_ratio=train_ratio, global_restriction = global_restriction, 
        sample_restriction = sample_restriction, sample_params = sample_params)
    
    N_c = ds_train.N_c # Number of channels
    Num_all_recon_pts = len(ds_train.recon_pool) # Number of points in the computational field

    def collate(batch): return default_collate(batch)
    common_kwargs = dict(batch_size=batch_size,
                         pin_memory=True,
                         persistent_workers=False,
                         prefetch_factor=4,
                         collate_fn=collate)

    ld_train = DataLoader(ds_train, shuffle=True,
                          num_workers=workers, 
                         **common_kwargs)
    ld_test  = DataLoader(ds_test,  shuffle=False,
                          num_workers=workers, 
                        **common_kwargs)
    
    print(f'Total channels of field data is {N_c}\n')
    return ld_train, ld_test, N_c, Num_all_recon_pts
