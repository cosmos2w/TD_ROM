
"""
evaluate.py
===========

This script performs evaluation of a trained TD-ROM model.
It follows these procedures:
  * Define the case index.
  * Search in the Save_config_files folder for the matching YAML config file 
    (ignoring the date portion in the filename).
  * Load the processed HDF5 dataset and extract the input tensors.
  * Build the model with hyperparameters from the YAML config.
  * Load the checkpoint for the given case.
  * Run the model to create a reconstructed spatiotemporal distribution.
  * Compute the MSE loss and plot/save the ground truth, prediction, and error fields.

"""

import os
import glob
import argparse
import pathlib
import yaml
import h5py
import torch
import datetime
import numpy as np

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.tri as mtri
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import animation

from typing import Sequence
from src.utils.plot_utils import save_plot
from src.utils.SpecialLosses import get_loss, select_high_freq_sensors
from src.models import (
    MLP, TD_ROM, TD_ROM_Bay_DD,
    FourierTransformerSpatialEncoder, DomainAdaptiveEncoder, 
    TemporalDecoderLinear, DelayEmbedNeuralODE, TemporalDecoderSoftmax, UncertaintyAwareTemporalDecoder, TemporalDecoderHierarchical,
    PerceiverReconstructor, SoftDomainAdaptiveReconstructor
)

device = torch.device("cpu")

def load_yaml_config(base_dir, net_index, stage_index, Repeat_id):
    """
    Search in the specified base_dir (e.g., project_root/Save_config_files)
    for a YAML config file corresponding to the given case_index.
    The file should have a name like "config_TDROM_idx_{net_index}_<date>.yaml" 
    (ignoring the date part).
    """
    # Construct pattern using the base directory
    pattern = os.path.join(str(base_dir), f"config_TDROM_idx_{net_index}_st_{stage_index}_num_{Repeat_id}*.yaml")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No YAML config file found for case_index {net_index} in {base_dir}."
        )
    print('we are here')
    print("test", sorted(files))
    cfg_file = sorted(files)[-1]  # choose the latest file if several are found
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"#-----------------------------------------\n"
          f"Loaded configuration from {cfg_file}\n"
          f"#-----------------------------------------")
    return cfg

def regenerate_equal_grid(Nx, Ny, delta_xy=0.01):
    """
    Create an equal-distance grid with spacing delta_xy, starting at 0.
    Returns:
      coords_u: (Nx_u, Ny_u, 2)
      xy_u    : (Nx_u*Ny_u, 2)
      x_u, y_u: 1D axes
    """
    Nx_u, Ny_u = Nx, Ny
    x_u = torch.arange(Nx_u, dtype=torch.float32) * delta_xy
    y_u = torch.arange(Ny_u, dtype=torch.float32) * delta_xy

    # Mesh (xy indexing: first dim is x, second is y)
    X_u, Y_u = torch.meshgrid(x_u, y_u, indexing="xy")
    coords_u = torch.stack([X_u, Y_u], dim=-1)  # (Nx_u, Ny_u, 2)
    xy_u = coords_u.reshape(-1, 2)
    return coords_u, xy_u, x_u, y_u

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] 
DATA_ROOT = REPO_ROOT 
def load_h5_data(h5_path):
    """
    Load the processed HDF5 dataset using the repository-standard format.

        fields      : Tensor of shape [B, N_t, N_x, N_y, N_z, F], F = 2: u & v
        coords      : Tensor of shape [N_x, N_y, N_z, N_dim]      (x,y)
        time_vector : Tensor of shape (N_t,)

    """

    h5_path = pathlib.Path(h5_path)
    if not h5_path.is_absolute():
        h5_path = (REPO_ROOT / h5_path).resolve()  # or use DATA_ROOT if fp is relative to repo root
    if not h5_path.exists():
        raise FileNotFoundError(f"Data file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        fields_np  = f["fields"][...].astype("float32")        # (1,N_t,N_x,N_y,N_z,2)
        coords  = f["coordinates"][...].astype("float32")   # (N_x,N_y,N_z,N_z,3)
        t_vec   = f["time"][...].astype("float32")          # (N_t,)
        if "conditions" in f.keys():
            U   = f["conditions"][...].astype("float32")
        else:
            U   = np.zeros((fields_np.shape[0], 1), np.float32)

        # optional original mean/std
        mean_np = f["orig_mean"][...].astype("float32") if "orig_mean" in f else None
        std_np  = f["orig_std"][...].astype("float32")  if "orig_std"  in f else None

    #UN-STANDARDIZE (if available)
    if mean_np is not None and std_np is not None:
        # mean_np (1, N_x, N_y, F) / std_np (1, 1, 1, F) / fields_np (1, N_t, N_x, N_y, N_z, F)
        # we need mean in shape (1, 1, N_x, N_y, 1, F), std_np will broadcast correctly as-is
        Nx, Ny, F = mean_np.shape[1], mean_np.shape[2], mean_np.shape[3]
        mean_np = mean_np.reshape((1, 1, Nx, Ny, 1, F))
        fields_np = fields_np * std_np + mean_np

    # reshape, normalise, flatten once -----------------------------------
    fields = torch.from_numpy(fields_np).squeeze(4)        # (B,N_t,N_x,N_y, 2)

    coords = torch.from_numpy(coords).squeeze(2)[..., :2]  # (N_x,N_y,2)
    Nx, Ny = coords.shape[:2]
    xy = coords.view(-1, 2)                                # (N_pts,2)
    print(f'In original dataset, xy is {xy}')

    t_vec = torch.from_numpy(t_vec)                        # (N_t,)

    return fields, xy, t_vec, torch.from_numpy(U)

# ---------------------------------------------------------
def build_model(cfg: dict,  N_c: int) -> TD_ROM:

    mlp_A2U     = [cfg["F_dim"], 256, 256, 256, cfg["U_dim"]]
    NN_A2U      = MLP(mlp_A2U)

    Net_Name = f"TD_ROM_id{cfg['case_index']}_st{cfg['Stage']}_num{cfg['Repeat_id']}"

    Use_Adaptive_Selection = cfg.get("Use_Adaptive_Selection", False)
    domain_decompose       = cfg.get('domain_decompose', False)
    CalRecVar              = cfg.get("CalRecVar", False)
    retain_cls             = cfg.get("retain_cls", False)
    Use_imp_in_dyn         = cfg.get("Use_imp_in_dyn", False)

    if domain_decompose and cfg['pooling'] == 'none':
        encoder = DomainAdaptiveEncoder(
            All_dim         = cfg["F_dim"],
            num_heads       = cfg["num_heads"],
            latent_layers   = cfg["num_layers"],
            N_channels      = N_c,
            num_freqs       = cfg["num_freqs"],
            latent_tokens   = cfg["latent_tokens"],
            pooling         = cfg["pooling"],
            retain_cls      = retain_cls,
        )
        print(f'\nBuilding Domain Adaptive Encoder as the sensor encoder!\n')
    else:
        encoder = FourierTransformerSpatialEncoder(
            All_dim         = cfg["F_dim"],
            num_heads       = cfg["num_heads"],
            latent_layers   = cfg["num_layers"],
            N_channels      = N_c,
            num_freqs       = cfg["num_freqs"],
            use_temporal    = cfg["use_temporal"],
            latent_tokens   = cfg["latent_tokens"],
            pooling         = cfg["pooling"],
        )
        print(f'\nBuilding FourierTransformerSpatialEncoder as the sensor encoder!\n')

    assert cfg["decoder_type"] in ("LinTrans", "DelayNODE", "CausalTrans", "UD_Trans")
    if cfg["decoder_type"] == "LinTrans":
        decoder_lat = TemporalDecoderLinear(
            d_model = cfg["F_dim"],
            n_layers= cfg["num_layers_propagator"],
            n_heads = cfg["num_heads"],
            dt      = cfg["delta_t"],
        )
    elif cfg["decoder_type"] == "DelayNODE":
        decoder_lat = DelayEmbedNeuralODE(
            d_model     = cfg["F_dim"],
            N_window    = cfg["N_window"],
            hidden_dims = cfg["hidden_dims"],
            dt          = cfg["delta_t"],
        )
    elif cfg["decoder_type"] == "CausalTrans":
        decoder_lat = TemporalDecoderSoftmax(
            d_model = cfg["F_dim"],
            n_layers= cfg["num_layers_propagator"],
            n_heads = cfg["num_heads"],
            dt      = cfg["delta_t"],
        )
    elif cfg["decoder_type"] == "UD_Trans": # UD = uncertainty-driven
        decoder_lat = TemporalDecoderHierarchical(
            d_model = cfg["F_dim"],
            n_layers= cfg["num_layers_propagator"],
            n_heads = cfg["num_heads"],
            dt      = cfg["delta_t"],
        )
        print(f'Building up the TemporalDecoderHierarchical for dynamic forecasting ! ')

    if domain_decompose and cfg['pooling'] == 'none':
        field_dec = SoftDomainAdaptiveReconstructor(
            d_model=cfg["F_dim"],
            num_heads=cfg["num_heads"],
            N_channels=N_c,
            # pe_module=encoder.embed['pos_embed'],
            
            importance_scale=cfg["importance_scale"],
            bandwidth_init=cfg["bandwidth_init"], top_k=cfg["top_k"], per_sensor_sigma=cfg["per_sensor_sigma"], 
            CalRecVar = CalRecVar,
            retain_cls = retain_cls,
        )
        print(f'\nBuilding SoftDomainAdaptiveReconstructor as the field reconstructor!\n')
    else:
        field_dec = PerceiverReconstructor(
                d_model  = cfg["F_dim"],          
                num_heads  = cfg["num_heads"],
                N_channels = N_c,
                pe_module  = encoder.pos_embed,   # weight sharing
            )
        print(f'\nBuilding Perceiver-style decoder as the field reconstructor!\n')

    if cfg["Use_Adaptive_Selection"] == True:
        net = TD_ROM_Bay_DD(cfg, encoder, decoder_lat, field_dec,
                    delta_t = cfg["delta_t"], N_window = cfg["N_window"], stage=cfg["Stage"],
                    use_adaptive_selection = Use_Adaptive_Selection, CalRecVar = CalRecVar, 
                    retain_cls = retain_cls, Use_imp_in_dyn = Use_imp_in_dyn)
    else:
        net = TD_ROM(encoder, decoder_lat, field_dec, 
                    delta_t = cfg["delta_t"], N_window = cfg["N_window"], stage=cfg["Stage"])  

    return net, Net_Name


# --------------------------------------------------------------- 1. Load Data

def load_data(cfg, args):
    torch.manual_seed(args.seed)
    fields, coords, time_vec, conditions = load_h5_data(cfg["data_h5"])
    B, T_total, N_x, N_y, N_c_tot = fields.shape
    N_pts = N_x * N_y
    return fields, coords, time_vec, conditions, B, T_total, N_pts, N_c_tot

# ------------------------------------------------------------ 2. Select Channels
def select_channels(cfg, args, N_c_tot, B):
    sel_channels = list(range(N_c_tot)) if cfg["channel"] == -1 else [cfg["channel"]]
    N_c = len(sel_channels)
    print(f'channel is {cfg["channel"]}, so N_c is {N_c}\n')

    Data_case_idx = args.Data_case_idx
    if not (0 <= Data_case_idx < B):
        print(f'Data_case_idx is {Data_case_idx}, B is {B}')
        raise ValueError(f"case_idx must be in 0…{B-1}")
    return sel_channels, N_c, Data_case_idx

# ------------------------------------------------------------ 3. Time Span
def validate_time_span(args, T_total):
    T_ini, N_pred = args.T_ini, args.N_pred
    T_end = T_ini + N_pred
    if T_end > T_total:
        raise ValueError(f"T_end={T_end} exceeds #time steps ({T_total}).")
    return T_ini, T_end, N_pred

# -------------------------------------------------------------- 4. Grid and Regions
def prepare_grid(args, coords, fields, T_ini, T_end, sel_channels, N_pts, N_c, cfg):

    xy_min, xy_max = coords.min(0).values, coords.max(0).values
    xy_norm = (coords - xy_min) / (xy_max - xy_min)  # Normalize coords to [0, 1]
    xy_norm = xy_norm * 2 - 1.0 # Now converted to (-1, 1)

    sample_restriction = bool(cfg['sample_restriction'])
    sample_params      = cfg["Sample_Parameters"]   # empty dict if None
    if sample_restriction:
        print(f'Using sample_restriction!\n')
        x_lo = sample_params.get("x_lo", 0.0)
        x_hi = sample_params.get("x_hi", 1.0)
        y_lo = sample_params.get("y_lo", 0.0)
        y_hi = sample_params.get("y_hi", 1.0)
        print(f'In restricted region, x_lo is {x_lo}, x_hi is {x_hi}, y_lo is {y_lo}, y_hi is {y_hi}')
    else:                            # no spatial masking requested
        x_lo, y_lo = -float("inf"), -float("inf")
        x_hi, y_hi =  float("inf"),  float("inf")

    # x_lo, x_hi = 0.30, 0.70
    # y_lo, y_hi = 0.10, 0.60

    mask = ((xy_norm[:, 0] >= x_lo) & (xy_norm[:, 0] <= x_hi) &
            (xy_norm[:, 1] >= y_lo) & (xy_norm[:, 1] <= y_hi))
    region_idx = torch.nonzero(mask, as_tuple=True)[0]  # Indices of region points
    print(f'region_idx.shape is {region_idx.shape}')
    print(f'region_idx is {region_idx}')

    global_restriction = cfg['global_restriction']
    recon_pool = region_idx if global_restriction else torch.arange(coords.size(0))
    recon_idx = torch.sort(recon_pool)[0]

    u_true_full = fields[args.Data_case_idx, T_ini:T_end, :, :, sel_channels].reshape(args.N_pred, N_pts, N_c) # (Nt, Npts, Nc)

    return xy_norm, region_idx, recon_idx, u_true_full

# ------------------------------------------------- 5. Preprocessing
def preprocess_data(cfg, fields, sel_channels):
    p_mode = cfg.get("process_mode", "None")
    eps = 1e-8
    stats = {}

    all_u = fields[..., sel_channels].reshape(-1, len(sel_channels))
    fwd, inv = None, None

    if p_mode == "MinMaxNorm":
        stats["mins"] = all_u.min(0).values
        stats["maxs"] = all_u.max(0).values
        fwd = lambda u: (u - stats["mins"]) / (stats["maxs"] - stats["mins"] + eps)
        inv = lambda u: u * (stats["maxs"] - stats["mins"]) + stats["mins"]

    elif p_mode == "MeanStdStand":
        stats["means"] = all_u.mean(0)
        stats["stds"] = all_u.std(0, unbiased=False)
        fwd = lambda u: (u - stats["means"]) / (stats["stds"] + eps)
        inv = lambda u: u * stats["stds"] + stats["means"]

    elif p_mode == "None":
        fwd = inv = lambda u: u
    else:
        raise ValueError(f"unknown process_mode '{p_mode}'")

    print(f'The pre-process model is {p_mode}')
    return fwd, inv, stats

# -------------------------------------------------------------- 6. Build Tensors
def build_tensors(u_true_full, xy_norm, region_idx, recon_idx, time_vec, 
                  T_ini, N_pred, N_window, args, obs_idx_in):
    if N_window > N_pred:
        raise ValueError("N_window larger than prediction horizon.")

    obs_idx = torch.sort(obs_idx_in)[0]               # (Ns,)
    Ns = obs_idx.numel()
    Nr = recon_idx.numel()

    t_slice = time_vec[T_ini:(T_ini + N_pred)]
    t_obs = t_slice[:N_window]

    # gather helpers ---------------------------------------------------------
    def gather(u, s_idx):          # u: (Nt,Npts,Nc)
        return u[:, s_idx, :]      # -> (Nt,|s_idx|,Nc)
    # print(f'recon_idx is {recon_idx}')
    # print(f'u_true_full.shape is {u_true_full.shape}')

    G_full_u = gather(u_true_full, recon_idx)
    # G_down_u = gather(u_true_full[:N_window], obs_idx)
    G_down_u = gather(u_true_full, obs_idx)

    xy_recon = xy_norm[recon_idx]
    xy_obs = xy_norm[obs_idx]

    def build_cube(u_tensor, y, t_vec):
        T, Ns = u_tensor.shape[:2]
        y2 = y.unsqueeze(0).expand(T, -1, -1)
        t2 = t_vec.view(-1, 1, 1).expand(-1, Ns, 1)
        return torch.cat((y2, u_tensor, t2), dim=-1)

    # G_down = build_cube(G_down_u, xy_obs, t_obs)
    G_down = build_cube(G_down_u, xy_obs, t_slice)
    return G_down.unsqueeze(0), G_full_u.unsqueeze(0), xy_recon, xy_obs, t_slice

# ------------------------------------------------- 7. Load Network
def load_network(cfg, N_c, device):
    model, Net_Name = build_model(cfg, N_c)
    print(f'The Net_Name is {Net_Name}')
    ckpt_path = os.path.join(cfg["save_net_dir"], f"Net_{Net_Name}.pth")
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if not (k.endswith('.S') or k.endswith('.Z'))}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

# -------------------------------------------------------------- 8. Inference
def perform_inference(model, G_down, G_full, coords_, U_vec, inv, CalRecVar):
    print(f'G_down.shape is {G_down.shape}')
    print(f'G_full.shape is {G_full.shape}')
    print(f'coords_.shape is {coords_.shape}')
    print(f'U_vec.shape is {U_vec.shape}')
    with torch.no_grad():
        if CalRecVar == True:
            pred_n, var_n, *_ = model(G_down, G_full, coords_, U_vec, teacher_force_prob=0.0)
            print(f'var_n.shape is {var_n.shape}')
            return inv(pred_n.squeeze(0).cpu()), var_n.squeeze(0).cpu
        else:
            pred_n, *_ = model(G_down, G_full, coords_, U_vec, teacher_force_prob=0.0)
            return inv(pred_n.squeeze(0).cpu())

# -------------------------------------------------------------- 9. Evaluation
def evaluate(u_true_phys, u_pred_phys, recon_idx):
    u_true_recon = u_true_phys[:, recon_idx, :]
    
    # Compute MSE (as before)
    mse_ch = torch.mean((u_true_recon - u_pred_phys) ** 2, dim=(0, 1))
    global_mse = float(mse_ch.mean())
    print(f"Global MSE = {global_mse:.4e}")
    for c, v in enumerate(mse_ch):
        print(f"Channel {c:>2d}: MSE = {float(v):.4e}")
    
    # Compute Absolute L2 Norm
    diff = u_true_recon - u_pred_phys
    l2_abs_ch = torch.sqrt(torch.sum(diff ** 2, dim=(0, 1)))
    global_l2_abs = float(torch.sqrt(torch.sum(diff ** 2)))
    print(f"Global Absolute L2 Norm = {global_l2_abs:.4e}")
    for c, v in enumerate(l2_abs_ch):
        print(f"Channel {c:>2d}: Absolute L2 Norm = {float(v):.4e}")
    
    # Compute Relative L2 Norm (as in the paper: ||error||_2 / ||true||_2)
    true_norm_ch = torch.sqrt(torch.sum(u_true_recon ** 2, dim=(0, 1)))
    global_true_norm = torch.sqrt(torch.sum(u_true_recon ** 2))
    epsilon = 1e-10  # Small value to avoid division by zero
    l2_rel_ch = l2_abs_ch / (true_norm_ch + epsilon)
    global_l2_rel = global_l2_abs / (global_true_norm + epsilon)
    print(f"Global Relative L2 Norm = {float(global_l2_rel):.4e}")
    for c, v in enumerate(l2_rel_ch):
        print(f"Channel {c:>2d}: Relative L2 Norm = {float(v):.4e}")
    
    return global_mse, global_l2_abs, global_l2_rel

# -------------------------------------------------------------- 10. Plot Results
def plot_results(u_true_phys, u_pred_phys, recon_idx, xy_recon, xy_sensors, time_vec, args, cfg, global_mse):

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    show_ids = args.plot_ids if args.plot_ids else (0, len(time_vec) // 2, len(time_vec) - 1)
    out_dir = pathlib.Path(cfg["save_recon_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    N_window    = cfg["N_window"]

    for c in range(u_true_phys.size(-1)):
        ch_dir = out_dir / f"Case_{args.Data_case_idx}_{ts}" / f"ch{c}"
        ch_dir.mkdir(parents=True, exist_ok=True)

        u_true_recon_np = u_true_phys[:, recon_idx, c].cpu().numpy()
        u_pred_recon_np = u_pred_phys[..., c].cpu().numpy()

        save_plot(
            u_true_recon_np,                 # (Nt, Nr)
            u_pred_recon_np,                 # (Nt, Nr)
            xy_recon.cpu().numpy(),          # (Nr, 2)
            time_vec.cpu().numpy(),          # (Nt,)
            timesteps=show_ids,
            out_dir=ch_dir,
            sensor_coords=xy_sensors.cpu().numpy(),
            cmap_field=args.cmap,
            cmap_err="inferno",
            dpi=150,
            N_window = N_window,
        )

    if args.N_pred > 1:
        # Add time series plots for designated points
        designated_points = [(0.5, 0.1), (0.6, 0.2), (0.4, 0.3)]  # Example designated points; can be adjusted
        xy_recon_np = xy_recon.cpu().numpy()  # (Nr, 2)

        for c in range(u_true_phys.size(-1)):
            ch_dir = out_dir / f"Case_{args.Data_case_idx}_{ts}" / f"ch{c}"
            ch_dir.mkdir(parents=True, exist_ok=True)

            u_true_recon_np = u_true_phys[:, recon_idx, c].cpu().numpy()  # (Nt, Nr)
            u_pred_recon_np = u_pred_phys[..., c].cpu().numpy()  # (Nt, Nr)
            time_np = time_vec.cpu().numpy()  # (Nt,)

            for i, point in enumerate(designated_points):
                # Find closest index
                dist = np.linalg.norm(xy_recon_np - np.array(point), axis=1)
                closest_idx = np.argmin(dist)

                true_ts = u_true_recon_np[:, closest_idx]
                pred_ts = u_pred_recon_np[:, closest_idx]

                # Plot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(time_np, true_ts, 'b-', label='Ground Truth')
                ax.plot(time_np, pred_ts, 'r--', label='Prediction')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'Value at {point}')
                ax.set_title(f'Channel {c} at point {point}')
                ax.legend()
                plt.savefig(ch_dir / f'time_series_point{i}_ch{c}.png', dpi=150)
                plt.close(fig)

    if args.SAVE_GIF and args.N_pred > 1:
        print("Saving GIFs for reconstructed temporal data...")
        for c in range(u_true_phys.size(-1)):
            ch_dir = out_dir / f"Case_{args.Data_case_idx}_{ts}" / f"ch{c}"
            ch_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract data for this channel
            u_true_recon_np = u_true_phys[:, recon_idx, c].cpu().numpy()  # (Nt, Nr)
            u_pred_recon_np = u_pred_phys[..., c].cpu().numpy()           # (Nt, Nr)
            xy = xy_recon.cpu().numpy()                                   # (Nr, 2)
            times = time_vec.cpu().numpy()                                # (Nt,)
            sensor_coords = xy_sensors.cpu().numpy() if xy_sensors is not None else None
            
            # ------------- reshape / sanity checks (adapted from save_plot) ---------------------------------
            u_true = np.asarray(u_true_recon_np)
            u_pred = np.asarray(u_pred_recon_np)
            X = np.asarray(xy)
            times_arr = np.asarray(times)
            
            if u_true.shape != u_pred.shape:
                raise ValueError("u_true and u_pred must have identical shape.")
            
            if u_true.ndim == 3:
                Nt, Nx, Ny = u_true.shape
                u_true = u_true.reshape(Nt, -1)
                u_pred = u_pred.reshape(Nt, -1)
            elif u_true.ndim == 2:
                Nt, Npts = u_true.shape
            else:
                raise ValueError("u_true/u_pred must have 2 or 3 dimensions.")
            
            if X.shape[0] != u_true.shape[1]:
                raise ValueError("Number of points in X does not match flattened field size.")
            if len(times_arr) != Nt:
                raise ValueError("`times` must have length Nt.")
            
            # ------------- triangulation (once) ------------------------------------
            u_true = np.ma.masked_invalid(u_true)
            u_pred = np.ma.masked_invalid(u_pred)
            
            x = X[:, 0]
            y = X[:, 1]
            triang = mtri.Triangulation(x, y)
            
            # Mask every triangle that touches at least one NaN vertex *in u_pred*
            bad_vertices = ~np.isfinite(u_pred[0])              # (Npts,)
            tri_mask = ~np.all(~bad_vertices[triang.triangles], axis=1)
            triang.set_mask(tri_mask)
            
            # ------------- colour limits (shared across all frames) -----------------
            field_min = np.minimum(u_true.min(), u_pred.min())
            field_max = np.maximum(u_true.max(), u_pred.max())
            
            err_all = np.abs(u_true - u_pred)
            err_min = err_all[err_all > 0].min() if np.any(err_all > 0) else 0.0
            err_max = err_all.max()
            
            # Function to update the frame for animation
            def update_frame(t_idx, fig, ax_true, ax_pred, ax_err, cbar_field, cbar_err):
                # Clear axes
                ax_true.clear()
                ax_pred.clear()
                ax_err.clear()
                
                u_t = u_true[t_idx]          # (Npts,)
                u_p = u_pred[t_idx]
                err = np.abs(u_t - u_p)
                mse = np.mean((u_t - u_p) ** 2)
                t_val = times_arr[t_idx]
                
                # Plot contours
                im_true = ax_true.tricontourf(
                    triang, u_t, levels=100, cmap=args.cmap,
                    vmin=field_min, vmax=field_max
                )
                im_pred = ax_pred.tricontourf(
                    triang, u_p, levels=100, cmap=args.cmap,
                    vmin=field_min, vmax=field_max
                )
                im_err = ax_err.tricontourf(
                    triang, err, levels=100, cmap="inferno",
                    vmin=err_min, vmax=err_max
                )
                
                # Mark sensors on ground truth panel
                if sensor_coords is not None:
                    ax_true.scatter(
                        sensor_coords[:, 0], sensor_coords[:, 1],
                        s=12, c="none", edgecolors="green", linewidths=1.2,
                        marker="o", zorder=4, label="sensors"
                    )
                    ax_true.legend(frameon=False, loc="upper right", fontsize=8)
                
                # Cosmetics
                ax_true.set_title("Ground truth")
                ax_pred.set_title("Prediction")
                ax_err.set_title("|Error|")
                
                for ax in (ax_true, ax_pred, ax_err):
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_aspect("equal")
                
                # Update suptitle
                fig.suptitle(f"t = {t_val:.3f}   |   MSE = {mse:.3e}", y=0.96)
                
                # Note: Colorbars are added once outside the animation, so no need to update here
            
            # Create figure for animation
            fig = plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.20, hspace=0.0)
            ax_true = fig.add_subplot(gs[0, 0])
            ax_pred = fig.add_subplot(gs[0, 1])
            ax_err = fig.add_subplot(gs[0, 2])
            
            # Initialize dummy plots for colorbar creation (using first frame)
            t_idx_init = 0
            u_t_init = u_true[t_idx_init]
            u_p_init = u_pred[t_idx_init]
            err_init = np.abs(u_t_init - u_p_init)
            
            im_true_init = ax_true.tricontourf(triang, u_t_init, levels=100, cmap=args.cmap, vmin=field_min, vmax=field_max)
            im_pred_init = ax_pred.tricontourf(triang, u_p_init, levels=100, cmap=args.cmap, vmin=field_min, vmax=field_max)
            im_err_init = ax_err.tricontourf(triang, err_init, levels=100, cmap="inferno", vmin=err_min, vmax=err_max)
            
            # Add colorbars once (shared)
            cbar_field = fig.colorbar(im_true_init, ax=[ax_true, ax_pred], shrink=0.9, pad=0.02)
            cbar_field.set_label("u")
            cbar_err = fig.colorbar(im_err_init, ax=ax_err, shrink=0.9, pad=0.02)
            cbar_err.set_label("|u - û|")
            
            # Clear initial plots after adding colorbars
            ax_true.clear()
            ax_pred.clear()
            ax_err.clear()
            
            # Create animation
            anim = animation.FuncAnimation(
                fig, update_frame, frames=Nt, 
                fargs=(fig, ax_true, ax_pred, ax_err, cbar_field, cbar_err), 
                interval=200
            )
            
            # Save as GIF
            gif_path = ch_dir / f"reconstructed_animation_ch{c}.gif"
            anim.save(gif_path, writer='imagemagick', fps=10, dpi=120)  # Adjust fps as needed
            plt.close(fig)
        
        print(f"GIFs saved to '{out_dir}'")

    print(f"All figures written to '{out_dir}'  (global MSE = {global_mse:.4e})")

def reconstruct(cfg, args):

    fields, coords, time_vec, conditions, B, T_total, N_pts, N_c_tot = load_data(cfg, args)

    sel_channels, N_c, Data_case_idx = select_channels(cfg, args, N_c_tot, B)

    T_ini, T_end, N_pred = validate_time_span(args, T_total)

    xy_norm, region_idx, recon_idx, u_true_full = prepare_grid(args, coords, fields, T_ini, T_end, sel_channels, N_pts, N_c, cfg)

    # -------------------------------------------------------------- 
    # random subset of region_idx with size = num_space_sample
    perm        = torch.randperm(len(region_idx))
    cand_idx    = torch.sort(region_idx[perm[:args.num_space_sample]])[0]   # (Ncand,)
    # ----------------------------------------------------------------

    fwd, inv, _ = preprocess_data(cfg, fields, sel_channels)

    model = load_network(cfg, N_c, device)

    u_true_full_n = fwd(u_true_full)
    Target_Window = cfg["N_window"]
    obs_idx_sel = cand_idx                                  # keep every candidate

    G_down, G_full, xy_recon, xy_obs, t_slice = build_tensors(u_true_full_n, xy_norm, region_idx, recon_idx, 
                                                              time_vec, T_ini, N_pred, Target_Window, args, obs_idx_in=obs_idx_sel)

    coords_recon = coords[recon_idx]
    coords_obs = coords[obs_idx_sel]

    G_down, G_full, xy_recon = G_down.to(device), G_full.to(device), xy_recon.to(device)
    U_vec = conditions[Data_case_idx:Data_case_idx + 1].to(device)

    CalRecVar = cfg.get("CalRecVar", False)
    if CalRecVar == True:
        u_pred_phys, var_phys = perform_inference(model, G_down, G_full, xy_recon.unsqueeze(0), U_vec, inv, CalRecVar)
    else:
        u_pred_phys = perform_inference(model, G_down, G_full, xy_recon.unsqueeze(0), U_vec, inv, CalRecVar)

    global_mse, global_l2_abs, global_l2_rel = evaluate(u_true_full, u_pred_phys, recon_idx)
    print(f'global_mse: {global_mse}, global_l2_abs: {global_l2_abs}, global_l2_rel: {global_l2_rel}')

    # plot_results(u_true_full, u_pred_phys, recon_idx, xy_recon, xy_sensors, t_slice, args, cfg, global_mse)
    plot_results(u_true_full, u_pred_phys, recon_idx, coords_recon, coords_obs, t_slice, args, cfg, global_l2_rel) 
    
def main():
    parser = argparse.ArgumentParser(description="TD-ROM Evaluation")

    parser.add_argument(
        "--dataset",
        default="collinear_flow_Re40",
        type=str,
        help="Datasets: channel_flow, collinear_flow_Re40, collinear_flow_Re100, cylinder_flow, FN_reaction_diffusion, sea_temperature, turbulent_combustion",
    )

    parser.add_argument('--indice', type=int, default=4, 
                        help='net checkpoint index: which net')
    parser.add_argument('--stage', type=int, default=0, 
                        help='net checkpoint index: which stage')
    parser.add_argument('--Repeat_id', type=int, default=0, 
                        help='Different propagator id sharing the same encoder and decoder')
    
    parser.add_argument("--Data_case_idx", type=int, default=0,
                        help="Case index to be selected for evaluation in the dataset")
    parser.add_argument("--T_ini", type=int, default=1000,
                        help="Initial time index from which to start prediction")
    parser.add_argument("--N_pred", type=int, default=1,
                        help="Number of time steps to predict")
    parser.add_argument("--num_space_sample", type=int, default=16,
                        help="Number of spatial points to supply to the encoder")
    
    parser.add_argument("--Select_Optimal", type=bool, default=False,
                        help="Decide whether we select best sensors based on retain probability")
    parser.add_argument("--Retain_Num", type=int, default=8,
                        help="Decide the number of topK important sensors")
    
    parser.add_argument("--SAVE_GIF", action='store_true', default=False,
                        help="If true and N_pred > 1, save a GIF of the reconstructed temporal data")
    parser.add_argument("--cmap", type=str, default="coolwarm",
                        help="Colormap for plotting the physical field")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--plot_ids", type=int, nargs="*", default=None,
                   help="indices within the prediction window to plot "
                        "(e.g. --plot_ids 0 50 75)")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset = args.dataset
    config_base_dir = project_root / "Save_config_files" / dataset / "config_bk_TDOM"

    cfg = load_yaml_config(config_base_dir, args.indice, args.stage, args.Repeat_id)

    reconstruct(cfg, args)

if __name__ == "__main__":
    main()

