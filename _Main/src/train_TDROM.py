
from __future__ import annotations
import argparse, datetime, pathlib, shutil, yaml, torch, csv, os, matplotlib, math
import sys
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F

from dataloading import make_loaders, make_loaders_DSUS

from utils.plot_utils import plot_loss_history
from utils.SpecialLosses import compute_psd, compute_spectrum

from models import (
    MLP, TD_ROM, TD_ROM_Bay_DD,
    FourierTransformerSpatialEncoder, DomainAdaptiveEncoder, 
    TemporalDecoderLinear, DelayEmbedNeuralODE, TemporalDecoderSoftmax, UncertaintyAwareTemporalDecoder, TemporalDecoderHierarchical,
    PerceiverReconstructor, SoftDomainAdaptiveReconstructor
)

# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="collinear_flow_Re40",
        type=str,
        help="Datasets: channel_flow, collinear_flow_Re40, collinear_flow_Re100, cylinder_flow, FN_reaction_diffusion, sea_temperature, turbulent_combustion",
    )
    p.add_argument(
        "--config",
        type=pathlib.Path,
        help="YAML config file (defaults to Save_config_files/<dataset>/YAML_config_TDROM.yaml)",
    )
    args = p.parse_args()

    if args.config is None:
        args.config = pathlib.Path(f"Save_config_files/{args.dataset}/YAML_config_TDROM.yaml")

    return args

# ---------------------------------------------------------
def load_cfg(yaml_path: pathlib.Path) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    # ---------- snapshot ----------
    time  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    idx = cfg.get("case_index", "X")   
    st  = cfg.get("Stage", "X") 
    num  = cfg.get("Repeat_id", "X")    

    snap = yaml_path.parent / "config_bk_TDOM" / f"config_TDROM_idx_{idx}_st_{st}_num_{num}_{time}.yaml"

    shutil.copyfile(yaml_path, snap)
    print(f"Now copied config → {snap}")

    cfg["psd_weight"] = cfg.get("psd_weight", 0.1)
    cfg["spectrum_weight"] = cfg.get("spectrum_weight", 0.1)

    return cfg

# ---------------------------------------------------------
def build_model(cfg: dict,  N_c: int) -> TD_ROM:

    mlp_A2U     = [cfg["F_dim"], 256, 256, 256, cfg["U_dim"]]
    NN_A2U      = MLP(mlp_A2U)

    Net_Name = f"TD_ROM_id{cfg['case_index']}_st{cfg['Stage']}_num{cfg['Repeat_id']}"
    channel_to_encode = cfg["channel_to_encode"] if "channel_to_encode" in cfg else None

    Use_Adaptive_Selection = cfg.get("Use_Adaptive_Selection", False)
    domain_decompose       = cfg.get('domain_decompose', False)
    CalRecVar              = cfg.get("CalRecVar", False)
    retain_cls             = cfg.get("retain_cls", False)
    retain_lat             = cfg.get("retain_lat", False)
    Use_imp_in_dyn         = cfg.get("Use_imp_in_dyn", False)

    if domain_decompose:
        encoder = DomainAdaptiveEncoder(
            All_dim         = cfg["F_dim"],
            num_heads       = cfg["num_heads"],
            latent_layers   = cfg["num_layers"],
            N_channels      = N_c,
            num_freqs       = cfg["num_freqs"],
            latent_tokens   = cfg["latent_tokens"],
            pooling         = cfg["pooling"],
            retain_cls      = retain_cls,
            retain_lat      = retain_lat,
            channel_to_encode = channel_to_encode
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

    if domain_decompose:
        field_dec = SoftDomainAdaptiveReconstructor(
            d_model=cfg["F_dim"],
            num_heads=cfg["num_heads"],
            N_channels=N_c,
            latent_tokens=cfg["latent_tokens"],
            # pe_module=encoder.embed['pos_embed'],
            
            importance_scale=cfg["importance_scale"],
            bandwidth_init=cfg["bandwidth_init"], top_k=cfg["top_k"], per_sensor_sigma=cfg["per_sensor_sigma"], 
            CalRecVar  = CalRecVar,
            retain_cls = retain_cls,
            retain_lat = retain_lat,
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

    if domain_decompose:
        net = TD_ROM_Bay_DD(cfg, encoder, decoder_lat, field_dec,
                    delta_t = cfg["delta_t"], N_window = cfg["N_window"], stage=cfg["Stage"],
                    use_adaptive_selection = Use_Adaptive_Selection, CalRecVar = CalRecVar, 
                    retain_cls = retain_cls, retain_lat = retain_lat, Use_imp_in_dyn = Use_imp_in_dyn)
        print(f'\nBuilding TD_ROM_Bay_DD as the model wrapper!\n')
    else:
        net = TD_ROM(encoder, decoder_lat, field_dec, 
                    delta_t = cfg["delta_t"], N_window = cfg["N_window"], stage=cfg["Stage"])  
        print(f'\nBuilding plain TD_ROM as the model wrapper!\n')

    return net, Net_Name

@torch.no_grad()
def select_and_gather_batch(G_d,                 # [B, T, N_x, N_call]
                            base_model,
                            epoch: int,
                            cfg,
                            downsample_ratio: float):
    """
    Selects a subset of sensors based on their importance scores (phi) and a dynamic ratio.

    Directly calculates the number of sensors to keep based on
    the `downsample_ratio` and uses `torch.topk` to select the most important ones
    according to their calculated `phi` values.
    """
    device            = G_d.device
    B, T, N_x, N_call = G_d.shape
    Stage             = cfg["Stage"]

    # --------------------------------------------------------
    # 1) Calculate phi values (importance scores) for all candidate sensors.
    #    This part of the logic remains the same as it's fundamental to your model.
    # --------------------------------------------------------
    coords_flat = G_d[:, 0, :, :2].reshape(-1, 2).contiguous()

    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(int(epoch))

        log_ab1 = base_model.phi_mlp_1(coords_flat)      # [BN_x, 2]
        alpha1  = torch.exp(log_ab1[:, 0]) + 1e-3
        beta1   = torch.exp(log_ab1[:, 1]) + 1e-3
        phi1    = torch.distributions.Beta(alpha1, beta1).rsample()

        if Stage == 1 and cfg["bayesian_phi"]["update_in_stage1"]:
            log_ab2 = base_model.phi_mlp_2(coords_flat)
            alpha2  = torch.exp(log_ab2[:, 0]) + 1e-3
            beta2   = torch.exp(log_ab2[:, 1]) + 1e-3
            phi2    = torch.distributions.Beta(alpha2, beta2).rsample()
        else:
            phi2    = torch.ones_like(phi1)

        phi_flat = (phi1 * phi2).view(B, N_x)            # [B, N_x]

    # --------------------------------------------------------
    # 2) SIMPLIFIED SELECTION LOGIC
    #    Directly select the top sensors based on the downsample_ratio.
    # --------------------------------------------------------
    # Calculate the number of sensors to keep for this batch.
    num_to_keep = max(1, int(N_x * downsample_ratio))

    # Directly select the indices of the top 'num_to_keep' sensors based on their phi values.
    _, top_indices = phi_flat.topk(k=num_to_keep, dim=1)

    # --------------------------------------------------------
    # 3) Gather the data from the selected sensors.
    # --------------------------------------------------------
    # Original shape: [B, num_to_keep]; Target shape:   [B, T, num_to_keep, N_call]
    # We need to expand it to match the dimensions of G_d for gathering along dim=2.
    idx_batch = top_indices.unsqueeze(1).unsqueeze(-1)      # Shape: [B, 1, num_to_keep, 1]
    idx_batch = idx_batch.expand(-1, T, -1, N_call)         # Shape: [B, T, num_to_keep, N_call]

    # Gather the data for the selected sensors. The size of dimension 2 is now `num_to_keep`.
    G_d_sel = torch.gather(G_d, dim=2, index=idx_batch)

    return G_d_sel, num_to_keep

# Helper for ELBO (orthogonalized from loop)
def compute_elbo(base_model, uncert, coords, stage, cfg):

    log_ab_1 = base_model.phi_mlp_1(coords)                         # [N_pts, 2]
    alpha_1  = torch.exp(log_ab_1[:, 0]) + 1e-3                     # [N_pts]
    beta_1   = torch.exp(log_ab_1[:, 1]) + 1e-3                     # [N_pts]
    phi_dist_1 = torch.distributions.Beta(alpha_1, beta_1)
    mean_phi_1 = alpha_1 / (alpha_1 + beta_1)  # [N_pts], in [0,1]
    mean_phi_1 = torch.clamp(mean_phi_1, min=1e-3, max=1-1e-3)  # Clamp for stability

    if stage == 1 and cfg["bayesian_phi"]["update_in_stage1"] == True:
        log_ab_2 = base_model.phi_mlp_2(coords)                         # [N_pts, 2]
        alpha_2  = torch.exp(log_ab_2[:, 0]) + 1e-3                     # [N_pts]
        beta_2   = torch.exp(log_ab_2[:, 1]) + 1e-3                     # [N_pts]
        phi_dist_2 = torch.distributions.Beta(alpha_2, beta_2)
        mean_phi_2 = alpha_2 / (alpha_2 + beta_2)  # [N_pts], in [0,1]
        mean_phi_2 = torch.clamp(mean_phi_2, min=1e-3, max=1-1e-3)  # Clamp for stability
    else:
        mean_phi_2 = torch.ones_like(mean_phi_1) # No temporal uncertainty considered

    phi_mean_all = mean_phi_1 * mean_phi_2

    mc_samples_elbo = cfg["bayesian_phi"].get("mc_samples_elbo", 5)
    log_lik = 0.0
    for _ in range(mc_samples_elbo):
        phi_1_s = phi_dist_1.rsample()  # [N_pts]
        phi_2_s = phi_dist_2.rsample() if 'phi_dist_2' in locals() else torch.ones_like(phi_1_s)
        phi_s = phi_1_s * phi_2_s

        log_lik += (uncert * phi_s).mean()
        # log_lik += (uncert * (1 - phi_s)).mean()
        # log_lik += - (uncert * (1 - phi_s)).mean()  # to penalize low phi in high-uncert areas

    log_lik /= mc_samples_elbo

    prior_dist      = torch.distributions.Beta(
        torch.full_like(alpha_1, cfg["bayesian_phi"]["prior_alpha"]),
        torch.full_like(beta_1, cfg["bayesian_phi"]["prior_beta"])
    )
    if stage == 0:
        kl              = cfg["bayesian_phi"]["lambda_kl"] * torch.distributions.kl.kl_divergence(phi_dist_1, prior_dist).mean()
        entropy         = phi_dist_1.entropy().mean()
    elif stage == 1 and cfg["bayesian_phi"]["update_in_stage1"] == True:
        kl              = cfg["bayesian_phi"]["lambda_kl"] * torch.distributions.kl.kl_divergence(phi_dist_2, prior_dist).mean()
        entropy         = phi_dist_2.entropy().mean()
    else:
        kl              = 0.0
        entropy         = 0.0

    var_phi = torch.var(phi_mean_all)
    var_weight = cfg["bayesian_phi"].get("var_weight", 1.0)  # Weight to encourage phi variance

    elbo = log_lik - kl + cfg["bayesian_phi"].get("vi_entropy_weight", 0.1) * entropy + var_weight * var_phi
    return -elbo  # Return negative for loss

# ---------------------------------------------------------------
def train(cfg: dict):

    device_ids = cfg["device_ids"]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    USE_DSUS = cfg.get("USE_DSUS", False)
    if USE_DSUS:
        train_ld, test_ld, N_c, Num_all_recon_pts = make_loaders_DSUS(
            cfg["data_h5"],
            num_time_sample    = cfg["num_time_sample"],
            num_space_sample   = cfg["num_space_sample"],

            Num_x              = cfg["Num_x"],
            Num_y              = cfg["Num_y"],
            global_downsample_ratio = cfg["global_downsample_ratio"],

            multi_factor       = cfg["multi_factor"],
            train_ratio        = cfg["train_ratio"],
            batch_size         = cfg["batch_size"],
            workers            = cfg["num_workers"],
            channel            = cfg["channel"],
            process_mode       = cfg["process_mode"],
            num_samples        = cfg["num_samples"],
            Full_Field_DownS   = cfg["Full_Field_DownS"],
            global_restriction = cfg["global_restriction"],
            sample_restriction = cfg["sample_restriction"],
            sample_params      = cfg["Sample_Parameters"],
        )
    else:
        train_ld, test_ld, N_c, Num_all_recon_pts = make_loaders(
            cfg["data_h5"],
            num_time_sample    = cfg["num_time_sample"],
            num_space_sample   = cfg["num_space_sample"],
            multi_factor       = cfg["multi_factor"],
            train_ratio        = cfg["train_ratio"],
            batch_size         = cfg["batch_size"],
            workers            = cfg["num_workers"],
            channel            = cfg["channel"],
            process_mode       = cfg["process_mode"],
            num_samples        = cfg["num_samples"],
            Full_Field_DownS   = cfg["Full_Field_DownS"],
            global_restriction = cfg["global_restriction"],
            sample_restriction = cfg["sample_restriction"],
            sample_params      = cfg["Sample_Parameters"],
        )

    # 2) model / opt ------------------------------------------
    Use_Adaptive_Selection = cfg.get("Use_Adaptive_Selection", False)  # Default to False if not in YAML
    CalRecVar              = cfg.get("CalRecVar", False)  if Use_Adaptive_Selection else False

    retain_cls             = cfg.get("retain_cls", False) 
    retain_lat             = cfg.get("retain_lat", False) 
    Supervise_Sensors      = cfg.get("Supervise_Sensors", False)

    BATCH_DOWNSAMPLE       = cfg.get("BATCH_DOWNSAMPLE", False)     # if we further downsample G_d in each batch for sparse adaptation test
    DOWNSAMPLE_LOGIC       = cfg.get("DOWNSAMPLE_LOGIC", "random")  # random or optimal. if optimal, use select_and_gather_batch()

    model, Net_Name = build_model(cfg, N_c)

    stage = cfg["Stage"]  
    Reload_Trained = cfg.get("Reload_Trained", False)
    if stage >= 1 or Reload_Trained is True:  
        load_Net_Name = f"TD_ROM_id{cfg['case_index']}_st0_num0"
        ckpt_path = os.path.join(cfg["save_net_dir"], f"Net_{load_Net_Name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Pre-trained stage-0 network not found at {ckpt_path}. Proceeding without loading.")
        else:
            ckpt = torch.load(ckpt_path)
            if "state_dict" in ckpt: sd = ckpt["state_dict"]
            else: sd = ckpt

            EXCLUDE_PREFIXES = (
                "decoder_lat.",
                "temporaldecoder.",
            )
            filtered_sd = {
                k: v
                for k, v in sd.items()
                if not any(k.startswith(pfx) for pfx in EXCLUDE_PREFIXES)
            }

            model.load_state_dict(filtered_sd, strict=False)

            #  re-init fresh temporal decoder
            def _init_temporal_decoder(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            model.temporaldecoder.apply(_init_temporal_decoder)
            print("Re-initialized temporal decoder according to current YAML.")

    # Freeze parameters based on stage
    if stage == 0:  
        for param in model.temporaldecoder.parameters():
            param.requires_grad = False
        if Use_Adaptive_Selection:
            for p in model.phi_mlp_2.parameters():       p.requires_grad = False # Do not update temporal uncrtainty
    elif stage == 1 or stage == 2:
        for p in model.fieldencoder.parameters():        p.requires_grad = False
        for p in model.decoder.parameters():             p.requires_grad = False
        if Use_Adaptive_Selection:
            for p in model.phi_mlp_1.parameters():       p.requires_grad = False # Do not update spatial uncrtainty

    # --- multi-GPU ---
    if len(device_ids) > 1:
        print(f"Using DataParallel on GPUs {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    def linear_warmup(epoch):
        if epoch < cfg["warmup_epochs"]:
            return epoch / cfg["warmup_epochs"]
        return 1.0  
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=linear_warmup)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.95, patience=500, min_lr=5e-5
    )

    mse = nn.MSELoss(reduction="mean")

    # 3) bookkeeping ------------------------------------------
    Case_num  = cfg["case_index"]
    loss_csv  = pathlib.Path(cfg["save_loss_dir"])
    loss_csv.mkdir(exist_ok=True, parents=True)
    case_dir  = loss_csv / f"Case{Case_num}"
    case_dir.mkdir(exist_ok=True, parents=True)

    csv_path  = case_dir / f"loss_log_{Net_Name}.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([f"created {datetime.datetime.now()}"])
        header = ['epoch', 'train_loss', 'train_loss_sum_mse']
        if N_c > 1:
            for ci in range(N_c):
                header.append(f'train_loss_mse_ch{ci}')
        header.extend(['train_loss_penalty', 'train_loss_TrajInfer', 'train_loss_UInfer', 'train_loss_KL', 'train_loss_NLL',  
                       'train_loss_psd', 'train_loss_spectrum',
                       'test_loss', 'test_loss_sum_mse'])
        if N_c > 1:
            for ci in range(N_c):
                header.append(f'test_loss_mse_ch{ci}')
        header.extend(['test_loss_penalty', 'test_loss_TrajInfer', 'test_loss_UInfer', 'test_loss_KL', 'test_loss_NLL',
                       'test_loss_psd', 'test_loss_spectrum'])
        csv.writer(f).writerow(header)

    best_test, epochs_no_imp = float("inf"), 0
    monitor_every = cfg["monitor_every"]
    patience_iter = cfg["patience_epochs"] // monitor_every

    train_hist, test_hist, train_hist_ch, test_hist_ch = [], [], [], []

    # ---------------------------------------------------------
    for epoch in range(1, cfg["num_epochs"] + 1):
        
        teacher_force_p = max(0.05, 0.50 - epoch / 1000)
        w_traj = 1.0
        w_cls  = cfg.get("Loss_cls_Weight", 0.10)

        if CalRecVar:
            nll_anneal_frac = min(epoch / cfg.get("nll_anneal_epochs", 100), 1.0)  
            nll_weight = cfg.get("nll_weight", 1.0) * nll_anneal_frac  

        # ----------------- TRAIN -----------------
        model.train()
        train_loss_list     = []
        train_loss_mse_list = []
        if N_c > 1:
            train_loss_mse_ch_list = [ [] for _ in range(N_c) ]
        train_loss_obs_list  = []
        train_loss_Traj_list = []
        train_loss_KL_list   = []
        train_loss_U_list    = []
        train_loss_nll_list  = [] 

        train_loss_psd_list = []
        train_loss_spectrum_list = []

        # for G_d, G_dt, G_f, Y, U in train_ld:
        #     G_d, G_dt, G_f, Y, U = map(lambda x: x.to(device), (G_d, G_dt, G_f, Y, U))
        #     B, num_time_sample, N_x, N_call = G_d.shape  

            # true_global_mean = G_f.mean(dim=2)  # [B, T, C]
            # true_global_var = G_f.var(dim=2)    # [B, T, C]
            # true_global_properties = torch.cat([true_global_mean, true_global_var], dim=-1) # [B, T, 2*C]

        for batch in train_ld:
            if USE_DSUS:
                # Unpack 6 tensors when using the DSUS loader
                G_d, G_dt, G_f, G_f_glb, Y, U = batch
                G_d, G_dt, G_f, G_f_glb, Y, U = map(
                    lambda x: x.to(device), 
                    (G_d, G_dt, G_f, G_f_glb, Y, U)
                )
            else:
                # Unpack 5 tensors when using the standard loader
                G_d, G_dt, G_f, Y, U = batch
                G_d, G_dt, G_f, Y, U = map(
                    lambda x: x.to(device), 
                    (G_d, G_dt, G_f, Y, U)
                )                

        # for G_d, G_dt, G_f, G_f_glb, Y, U in train_ld:
        #     G_d, G_dt, G_f, G_f_glb, Y, U = map(lambda x: x.to(device), (G_d, G_dt, G_f, G_f_glb, Y, U))

            B, num_time_sample, N_x, N_call = G_d.shape  

            downsample_ratio = 1.0
            if BATCH_DOWNSAMPLE:
                # Generate a random ratio for this specific batch in the range [0.50, 1.0].
                ratio = torch.rand(1).item() * 0.50 + 0.50
                downsample_ratio = ratio  # Store the ratio for loss scaling later.
                # print(f'downsample_ratio is {downsample_ratio}')

                if DOWNSAMPLE_LOGIC == "optimal":
                    G_d_sel, num_to_keep = select_and_gather_batch(G_d, base_model, epoch, cfg, downsample_ratio)
                    G_d = G_d_sel

                elif DOWNSAMPLE_LOGIC == "random":
                    # Calculate the number of spatial points to keep.
                    num_to_keep = max(1, int(N_x * ratio))
                    # Generate a random permutation of indices along the spatial dimension (N_x)
                    indices = torch.randperm(N_x, device=G_d.device)[:num_to_keep]
                    # Reshape and expand the indices to be compatible with torch.gather.
                    indices = indices.view(1, 1, num_to_keep, 1)
                    indices = indices.expand(B, num_time_sample, -1, N_call)
                    # 5. Create the downsampled subset of G_d by gathering along dimension 2.
                    G_d = torch.gather(G_d, dim=2, index=indices)
                else:
                    print(' Error in DOWNSAMPLE_LOGIC! Should be either "random" or "optimal" ')
                    exit()

            opt.zero_grad()
            out, out_logvar, obs, traj, traj_logvar, G_u_cls, G_u_mean_Sens, G_u_logvar_Sens = model(G_d, G_f, Y, U, teacher_force_p)

            # Compute NLL loss
            nll_loss = 0.0
            if CalRecVar and stage == 0:
                var = torch.exp(out_logvar) + 1e-6  # [B, T, P, C] variance (positive)
                var_sen = (torch.exp(G_u_logvar_Sens) + 1e-6).mean() if Supervise_Sensors==True and G_u_logvar_Sens != None else 0

                nll_main = 0.5 * (( (out - G_f)**2 / var ) + out_logvar).mean() + var_sen  # Mean over all dims
                nll_loss = nll_weight * (nll_main)

            # Compute PSD/spectrum losses
            if stage == 0:
                psd_out = compute_psd(out, Y[0])    # [B, n_bins]
                psd_f   = compute_psd(G_f, Y[0])    # [B, n_bins]
                # Log-space MSE for scale invariance
                l_psd = ((psd_out + 1e-8).log() - (psd_f + 1e-8).log()).pow(2).mean() * cfg["psd_weight"]
                l_spectrum = torch.tensor(0.0, device=device)
            elif stage == 1:
                l_psd    = torch.tensor(0.0, device=device)
                spec_out = compute_spectrum(out)    # [B, T//2+1]
                spec_f   = compute_spectrum(G_f)    # [B, T//2+1]
                l_spectrum = ((spec_out + 1e-8).log() - (spec_f + 1e-8).log()).pow(2).mean() * cfg["spectrum_weight"]
            else:
                l_psd = torch.tensor(0.0, device=device)
                l_spectrum = torch.tensor(0.0, device=device)

            # -----------------------------
            update_phi = Use_Adaptive_Selection and (
                stage == 0 or cfg["bayesian_phi"].get("update_in_stage1", True))
            
            if update_phi:  # ELBO components: to reward high phi on high residuals
                coords_for_phi = Y[0]  # [N_pts, 2]

                # ----------------------------------------
                # var_pred = torch.exp(out_logvar).detach() if out_logvar is not None else torch.zeros_like(out)
                # Choice (1) pool over batch/time/channel
                # uncert = var_pred.mean(dim=(0,1,3))            # shape [N_pts]

                # Choice (2) Hierarchical (per-batch mean, then max across batches)
                # uncert_batch = var_pred.mean(dim=(1,3))  # Mean over T/C per batch → [B, N_pts]
                # uncert = uncert_batch.max(dim=0)[0]      # Max over batches → [N_pts]

                # Choice (3) per-batch max & mean mixing across batches
                # uncert_max  = torch.amax(var_pred, dim=(0,1,3))      # Max over batches → [N_pts]
                # uncert_mean = var_pred.mean(dim=(0,1,3))             # Mean over batches → [N_pts]
                # uncert = 0.5 * uncert_max # + 0.5 * uncert_mean
                # ----------------------------------------

                # ----------------------------------------
                # residuals = (out - G_f).abs().mean([0,1,3])   # [B, N_t, N_pts, N_c] -> [N_pts]

                # residuals = (out - G_f).abs().mean([1,3])       # Mean over T/C per batch → [B, N_pts]
                # residuals = residuals.max(dim=0)[0]             # Max over batches → [N_pts]

                residuals = (out - G_f).abs()[0, 0, :, 0]

                uncert = residuals.detach()
                # ----------------------------------------

                uncert = (uncert - uncert.min()) / (uncert.max() - uncert.min() + 1e-8)  # Normalize
                # uncert = uncert ** 1.5

                # Compute spectral_uncert and blend with uncert
                if stage == 0  : l_spectral = l_psd
                elif stage == 1: l_spectral = l_spectrum
                else:
                    l_spectral = torch.tensor(0.0, device=device)
                
                # Blend in temporal uncertainty from latent traj_logvar (for mlp_phi_2 update)
                if stage == 1 and traj_logvar is not None:
                    temporal_uncert = traj_logvar.exp().mean(dim=(0,1,-1)).detach()  # Global scalar (mean var over B, T, D)
                    temporal_uncert = (temporal_uncert - temporal_uncert.min()) / (temporal_uncert.max() - temporal_uncert.min() + 1e-6)
                    temporal_blend_weight = cfg.get("temporal_uncert_weight", 0.5) * min(epoch / cfg.get("nll_anneal_epochs", 100), 1.0)  # Annealed
                    # uncert = (1 - temporal_blend_weight) * uncert + temporal_blend_weight * temporal_uncert  # Blend (broadcast scalar to [N_pts])
                    uncert = temporal_uncert

                # Per-point attribution (simple gradients; [N_pts])
                if l_spectral > 0:
                    spectral_uncert = torch.autograd.grad(l_spectral, out, retain_graph=True, create_graph=False, grad_outputs=torch.ones_like(l_spectral))[0]
                    spectral_uncert = spectral_uncert.abs().mean(dim=(0,1,3)).detach()  # [N_pts]
                    # Normalize with clamping to prevent explosion from large gradients
                    spectral_uncert = torch.clamp(spectral_uncert, min=1e-8, max=1e6)  # Prevent extremes
                    spectral_uncert = (spectral_uncert - spectral_uncert.min()) / (spectral_uncert.max() - spectral_uncert.min() + 1e-6)
                    spectral_uncert = spectral_uncert ** 2
                    # Weighted blend 
                    spectral_blend_weight = cfg.get("spectral_blend_weight", 0.5)
                    uncert = (1 - spectral_blend_weight) * uncert + spectral_blend_weight * spectral_uncert

                # ----------------------------------------
                elbo = compute_elbo(base_model, uncert, coords_for_phi, stage, cfg)
                lambda_elbo     = cfg["bayesian_phi"]["lambda_elbo"] * min(epoch / cfg["bayesian_phi"]["anneal_epochs"], 1.0)
                elbo_loss       = elbo * lambda_elbo
            else:
                elbo_loss = 0.0
            # -----------------------------

            if N_c > 1: # Vectorized per-channel MSE
                loss_mse_channels = [mse(out[..., ci], G_f[..., ci]) for ci in range(N_c)]
                loss_mse  = sum(loss_mse_channels)
            else:
                loss_mse  = mse(out, G_f)

            # loss_U = 0.0    
            # 1031 asign loss_U to supervise cls
            if retain_cls is True and G_u_cls is not None and G_f_glb is not None:
                if N_c > 1:
                    loss_mse_channels_cls = []
                    for ci in range(N_c):
                        loss_channel = mse(G_u_cls[..., ci], G_f[..., ci],)
                        loss_mse_channels_cls.append(loss_channel)
                    loss_U  = w_cls * sum(loss_mse_channels_cls)
                else:
                    loss_U  = w_cls * mse(G_u_cls, G_f_glb)
                # loss_U = w_cls * mse(G_u_cls, true_global_properties.detach())
            else: loss_U = 0.0

            if stage == 0 and Supervise_Sensors:
                # print(f'G_u_mean_Sens.shape is {G_u_mean_Sens.shape}')
                # print(f'G_d.shape is {G_d.shape}')
                # loss_obs = mse(G_u_mean_Sens, G_d[:, :, :, 2:3])

                if N_c > 1: # Vectorized per-channel MSE
                    loss_obs_channels = [mse(G_u_mean_Sens[..., ci], G_d[:, :, :, (2+ci)]) for ci in range(N_c)] 
                    loss_obs  = sum(loss_obs_channels)
                else:
                    loss_obs = mse(G_u_mean_Sens, G_d[:, :, :, 2:3])

            else:  loss_obs = 0.0
            
            if cfg["N_window"] == num_time_sample or stage == 0: loss_traj = 0.0
            else:
                loss_traj = w_traj * mse(traj[:, cfg["N_window"]:num_time_sample, :], obs[:, cfg["N_window"]:, :])
            
            loss_kl   = elbo_loss   # Use kl term to document elbo loss only in train loop

            loss = (loss_mse + loss_obs) * downsample_ratio + loss_traj + loss_kl + loss_U + nll_loss + l_psd + l_spectrum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss_list.append(loss )
            train_loss_mse_list.append(loss_mse )
            if N_c > 1:
                for ci, l in enumerate(loss_mse_channels):
                    train_loss_mse_ch_list[ci].append(l)
            train_loss_obs_list.append(loss_obs )
            train_loss_Traj_list.append(loss_traj )
            train_loss_KL_list.append(loss_kl )
            train_loss_U_list.append(loss_U )
            train_loss_nll_list.append(nll_loss)

            train_loss_psd_list.append(l_psd)
            train_loss_spectrum_list.append(l_spectrum)

        avg_train_loss          = sum(train_loss_list)      / len(train_loss_list)
        avg_train_loss_mse      = sum(train_loss_mse_list)  / len(train_loss_mse_list)
        if N_c > 1: avg_train_loss_mse_ch = [ sum(l)/len(l) for l in train_loss_mse_ch_list ]
        avg_train_loss_obs      = sum(train_loss_obs_list)  / len(train_loss_obs_list)
        avg_train_loss_Traj     = sum(train_loss_Traj_list) / len(train_loss_Traj_list)
        avg_train_loss_KL       = sum(train_loss_KL_list)   / len(train_loss_KL_list)
        avg_train_loss_U        = sum(train_loss_U_list)    / len(train_loss_U_list)
        avg_train_loss_nll      = sum(train_loss_nll_list)  / len(train_loss_nll_list)

        avg_train_loss_psd      = sum(train_loss_psd_list) / len(train_loss_psd_list)
        avg_train_loss_spectrum = sum(train_loss_spectrum_list) / len(train_loss_spectrum_list)

        if epoch < cfg["warmup_epochs"]:
            warmup_sched.step()
            current_lr = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch}, Learning rate: {current_lr}, Loss: {loss.item()}")
        elif epoch >= cfg["warmup_epochs"]:
            plateau_sched.step(loss)

        # ----------------- TEST -----------------
        model.eval()
        with torch.no_grad():
            test_loss_list = []
            test_loss_mse_list = []
            if N_c > 1:
                test_loss_mse_ch_list = [ [] for _ in range(N_c) ]
            test_loss_obs_list  = []
            test_loss_Traj_list = []
            test_loss_KL_list   = []
            test_loss_U_list    = []
            test_loss_nll_list  = []
            test_loss_psd_list = []
            test_loss_spectrum_list = []

            first_batch_viz_data = None 

            # for batch_idx, (G_d, G_dt, G_f, Y, U) in enumerate(test_ld):
            #     G_d, G_dt, G_f, Y, U = map(lambda x: x.to(device), (G_d, G_dt, G_f, Y, U))

            for batch_idx, batch in enumerate(test_ld):
                if USE_DSUS:
                    # Unpack 6 tensors when using the DSUS loader
                    G_d, G_dt, G_f, G_f_glb, Y, U = batch
                    G_d, G_dt, G_f, G_f_glb, Y, U = map(
                        lambda x: x.to(device), 
                        (G_d, G_dt, G_f, G_f_glb, Y, U)
                    )
                else:
                    # Unpack 5 tensors when using the standard loader
                    G_d, G_dt, G_f, Y, U = batch
                    G_d, G_dt, G_f, Y, U = map(
                        lambda x: x.to(device), 
                        (G_d, G_dt, G_f, Y, U)
                    )

            # for batch_idx, (G_d, G_dt, G_f, G_f_glb, Y, U) in enumerate(test_ld):
            #     G_d, G_dt, G_f, G_f_glb, Y, U = map(lambda x: x.to(device), (G_d, G_dt, G_f, G_f_glb, Y, U))

                B, num_time_sample, N_x, N_call = G_d.shape

                true_global_mean = G_f.mean(dim=2)  # [B, T, C]
                true_global_var = G_f.var(dim=2)    # [B, T, C]
                true_global_properties = torch.cat([true_global_mean, true_global_var], dim=-1) # [B, T, 2*C]

                out, out_logvar, obs, traj, traj_logvar, G_u_cls, G_u_mean_Sens,G_u_logvar_Sens = model(G_d, G_f, Y, U, teacher_force_p)

                nll_loss = 0.0
                if CalRecVar:
                    # For visualization, store data from first batch (batch_idx == 0)
                    if batch_idx == 0 and first_batch_viz_data is None:
                        # Store Y and logvar for first case (index 0) in this batch
                        # Compute variance from logvar (exp for positivity); mean over T and C for viz simplicity

                        out_logvar_ = out_logvar.contiguous().clone()

                        viz_variance = torch.exp(out_logvar_[0])
                        viz_variance = viz_variance.mean(dim=(0, 2)).detach().cpu()
                        viz_variance = viz_variance.numpy()  # [P] (mean variance per query point)

                        viz_Y = Y[0].detach().cpu().numpy()  # [P, 2/3] for first case
                        first_batch_viz_data = (viz_Y, viz_variance)

                    if stage == 0:
                        var = torch.exp(out_logvar) + 1e-6
                        var_sen = (torch.exp(G_u_logvar_Sens) + 1e-6).mean() if Supervise_Sensors==True and G_u_logvar_Sens != None else 0
                        nll_main = 0.5 * (( (out - G_f)**2 / var ) + out_logvar).mean() + var_sen
                        nll_loss = nll_weight * nll_main  # Unweighted for test logging

                if N_c > 1:
                    loss_mse_channels = [mse(out[..., ci], G_f[..., ci]) for ci in range(N_c)]
                    loss_mse  = sum(loss_mse_channels)
                else:
                    loss_mse  = mse(out, G_f)

                if retain_cls is True and G_u_cls is not None and G_f_glb is not None:
                    if N_c > 1:
                        loss_mse_channels_cls = []
                        for ci in range(N_c):
                            loss_channel = mse(G_u_cls[..., ci], G_f[..., ci],)
                            loss_mse_channels_cls.append(loss_channel)
                        loss_U  = w_cls * sum(loss_mse_channels_cls)
                    else:
                        loss_U  = w_cls * mse(G_u_cls, G_f_glb)
                    # loss_U = w_cls * mse(G_u_cls, true_global_properties.detach())
                else: loss_U = 0.0
                # loss_U = 0.0

                if stage == 0 and Supervise_Sensors:
                    # loss_obs = mse(G_u_mean_Sens, G_d[:, :, :, 2:3])

                    if N_c > 1: # Vectorized per-channel MSE
                        loss_obs_channels = [mse(G_u_mean_Sens[..., ci], G_d[:, :, :, (2+ci)]) for ci in range(N_c)] 
                        loss_obs  = sum(loss_obs_channels)
                    else:
                        loss_obs = mse(G_u_mean_Sens, G_d[:, :, :, 2:3])

                else:  loss_obs = 0.0

                if cfg["N_window"] == num_time_sample or stage == 0: loss_traj = 0.0
                else:
                    loss_traj = w_traj * mse(traj[:, cfg["N_window"]:num_time_sample, :], obs[:, cfg["N_window"]:, :])
                
                loss_kl = 0.0

                if stage == 0:
                    psd_out = compute_psd(out, Y[0])    # [B, n_bins]
                    psd_f   = compute_psd(G_f, Y[0])    # [B, n_bins]
                    # Log-space MSE for scale invariance
                    l_psd = ((psd_out + 1e-8).log() - (psd_f + 1e-8).log()).pow(2).mean() * cfg["psd_weight"]
                    l_spectrum = torch.tensor(0.0, device=device)
                elif stage == 1:
                    l_psd    = torch.tensor(0.0, device=device)
                    spec_out = compute_spectrum(out)    # [B, T//2+1]
                    spec_f   = compute_spectrum(G_f)    # [B, T//2+1]
                    l_spectrum = ((spec_out + 1e-8).log() - (spec_f + 1e-8).log()).pow(2).mean() * cfg["spectrum_weight"]
                else:
                    l_psd = torch.tensor(0.0, device=device)
                    l_spectrum = torch.tensor(0.0, device=device)

                loss = loss_mse + loss_obs + loss_traj + loss_kl + loss_U + l_psd + l_spectrum

                test_loss_list.append(loss )
                test_loss_mse_list.append(loss_mse )
                if N_c > 1:
                    for ci, l in enumerate(loss_mse_channels):
                        test_loss_mse_ch_list[ci].append(l)
                test_loss_obs_list.append(loss_obs )
                test_loss_Traj_list.append(loss_traj )
                test_loss_KL_list.append(loss_kl )
                test_loss_U_list.append(loss_U )
                test_loss_nll_list.append(nll_loss)
                test_loss_psd_list.append(l_psd)
                test_loss_spectrum_list.append(l_spectrum)

            avg_test_loss       = sum(test_loss_list)       / len(test_loss_list)
            avg_test_loss_mse   = sum(test_loss_mse_list)   / len(test_loss_mse_list)
            if N_c > 1:
                avg_test_loss_mse_ch   = [ sum(l)/len(l) for l in test_loss_mse_ch_list ]
            avg_test_loss_obs   = sum(test_loss_obs_list)   / len(test_loss_obs_list)
            avg_test_loss_Traj  = sum(test_loss_Traj_list)  / len(test_loss_Traj_list)
            avg_test_loss_KL    = sum(test_loss_KL_list)    / len(test_loss_KL_list)
            avg_test_loss_U     = sum(test_loss_U_list)     / len(test_loss_U_list)
            avg_test_loss_nll   = sum(test_loss_nll_list)   / len(test_loss_nll_list)  # REVISED: Average test NLL
            avg_test_loss_psd = sum(test_loss_psd_list) / len(test_loss_psd_list)
            avg_test_loss_spectrum = sum(test_loss_spectrum_list) / len(test_loss_spectrum_list)

        train_hist.append(avg_train_loss_mse.item())
        test_hist.append(avg_test_loss_mse.item())
        if N_c > 1:
            train_hist_ch.append(avg_train_loss_mse_ch)
            test_hist_ch.append(avg_test_loss_mse_ch)

        # ----------------- logging -----------------
        if epoch % monitor_every == 0:

            if N_c > 1:
                train_ch_str = ", ".join(
                    f"ch{c}:{avg_train_loss_mse_ch[c]:.6f}"
                    for c in range(N_c)
                )
                test_ch_str = ", ".join(
                    f"ch{c}:{avg_test_loss_mse_ch[c]:.6f}"
                    for c in range(N_c)
                )
                train_mse_summary = f"{avg_train_loss_mse:.6f} [{train_ch_str}]"
                test_mse_summary  = f"{avg_test_loss_mse:.6f} [{test_ch_str}]"
            else:
                train_mse_summary = f"{avg_train_loss_mse:.6f}"
                test_mse_summary  = f"{avg_test_loss_mse:.6f}"

            print(
                f"Epoch {epoch}:\n"
                f" Train Loss: {avg_train_loss:.6f} (MSE: {train_mse_summary}, "
                f"Penalty: {avg_train_loss_obs:.6f}, U_Traj: {avg_train_loss_Traj:.6f}, "
                f"U_infer: {avg_train_loss_U:.6f}, KL: {avg_train_loss_KL:.6f}, NLL: {avg_train_loss_nll:.6f},"
                f"PSD: {avg_train_loss_psd:.6f}, Spectrum: {avg_train_loss_spectrum:.6f})\n"
                f" Test  Loss: {avg_test_loss:.6f} (MSE: {test_mse_summary}, "
                f"Penalty: {avg_test_loss_obs:.6f}, U_Traj: {avg_test_loss_Traj:.6f}, "
                f"U_infer: {avg_test_loss_U:.6f}, KL: {avg_test_loss_KL:.6f}, NLL: {avg_test_loss_nll:.6f},"
                f"PSD: {avg_test_loss_psd:.6f}, Spectrum: {avg_test_loss_spectrum:.6f})"
            )

            with open(csv_path, "a", newline="") as f:
                row = [epoch, f"{avg_train_loss:.6f}", f"{avg_train_loss_mse:.6f}"]
                if N_c > 1:
                    for loss_ch in avg_train_loss_mse_ch:
                        row.append(f"{loss_ch:.6f}")
                row.extend([
                    f"{avg_train_loss_obs:.6f}", f"{avg_train_loss_Traj:.6f}", 
                    f"{avg_train_loss_U:.6f}", f"{avg_train_loss_KL:.6f}", f"{avg_train_loss_nll:.6f}", 
                    f"{avg_train_loss_psd:.6f}", f"{avg_train_loss_spectrum:.6f}",
                    f"{avg_test_loss:.6f}", f"{avg_test_loss_mse:.6f}"])
                if N_c > 1:
                    for loss_ch in avg_test_loss_mse_ch:
                        row.append(f"{loss_ch:.6f}")
                row.extend([
                    f"{avg_test_loss_obs:.6f}", f"{avg_test_loss_Traj:.6f}", 
                    f"{avg_test_loss_U:.6f}", f"{avg_test_loss_KL:.6f}", f"{avg_test_loss_nll:.6f}", 
                    f"{avg_test_loss_psd:.6f}", f"{avg_test_loss_spectrum:.6f}"
                ])
                csv.writer(f).writerow(row)

            # ---------------- Update dynamic loss figure ----------------

            plot_loss_history(epoch=epoch,
                            train_hist=train_hist,
                            test_hist=test_hist,
                            train_hist_ch=train_hist_ch,
                            test_hist_ch=test_hist_ch,
                            save_dir=cfg["save_loss_dir"],
                            net_num=cfg["case_index"],
                            net_name=Net_Name,
                            )

            if Use_Adaptive_Selection and epoch % cfg["bayesian_phi"]["viz_every"] == 0:

                viz_dir = cfg["bayesian_phi"]["viz_save_dir"]
                Case_num = cfg["case_index"]
                Stage_num = cfg["Stage"]
                Repeat_id = cfg["Repeat_id"]
                os.makedirs(f"{viz_dir}/Case_num{Case_num}_Stage{Stage_num}_num{Repeat_id}", exist_ok=True)

                coords = Y[0]  # [N_pts, 2] full coords

                # Compute Beta params from MLP on current points
                # Spatial contributions by phi_mlp_1:
                log_ab_1 = base_model.phi_mlp_1(coords)                         # [N_pts, 2]
                alpha_1  = torch.exp(log_ab_1[:, 0]) + 1e-3                     # [N_pts]
                beta_1   = torch.exp(log_ab_1[:, 1]) + 1e-3                     # [N_pts]
                # Compute mean phi (Beta expectation) instead of sampling
                mean_phi_1 = alpha_1 / (alpha_1 + beta_1)  # [N_pts], in [0,1]
                mean_phi_1 = torch.clamp(mean_phi_1, min=1e-3, max=1-1e-3)  # Clamp for stability
                mean_phi_1_np = mean_phi_1.detach().cpu().numpy()  # [N_pts]

                # Temporal contributions by phi_mlp_2:
                if stage == 1 and cfg["bayesian_phi"]["update_in_stage1"] == True:
                    log_ab_2 = base_model.phi_mlp_2(coords)                         # [N_pts, 2]
                    alpha_2  = torch.exp(log_ab_2[:, 0]) + 1e-3                     # [N_pts]
                    beta_2   = torch.exp(log_ab_2[:, 1]) + 1e-3                     # [N_pts]
                    mean_phi_2 = alpha_2 / (alpha_2 + beta_2)  # [N_pts], in [0,1]
                    mean_phi_2 = torch.clamp(mean_phi_2, min=1e-3, max=1-1e-3)  # Clamp for stability
                else:
                    mean_phi_2 = torch.ones_like(mean_phi_1) # No temporal uncertainty considered
                mean_phi_2_np = mean_phi_2.detach().cpu().numpy()  # [N_pts]
                mean_phi = mean_phi_1 * mean_phi_2
                mean_phi_np = mean_phi.detach().cpu().numpy()  # [N_pts]

                # Extract x, y for scatter (point cloud)
                x = coords[:, 0].detach().cpu().numpy()  # [N_pts]
                y = coords[:, 1].detach().cpu().numpy()  # [N_pts]
                
                residuals_np = uncert.detach().cpu().numpy()  # [N_pts] , or residuals.detach().cpu().numpy()
                # Residuals plot
                plt.figure(figsize=(6, 6))
                res_vmin = float( residuals_np.min() )
                res_vmax = float( residuals_np.max() )
                scatter_res = plt.scatter(x, y, c=residuals_np, cmap='plasma', vmin=res_vmin, vmax=res_vmax, s=10)
                plt.colorbar(scatter_res, label='Residuals')
                plt.title(f'Epoch {epoch}: Residuals Distribution')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(os.path.join(f"{viz_dir}/Case_num{Case_num}_Stage{Stage_num}_num{Repeat_id}", f'residuals_epoch_{epoch}.png'))
                plt.close()

                # Plot scatter by averaged phi_mean
                plt.figure(figsize=(6, 6))
                vmin = float(mean_phi.min().item())
                vmax = float(mean_phi.max().item())
                scatter = plt.scatter(x, y, c=mean_phi_np, cmap='viridis', vmin=vmin, vmax=vmax, s=10)
                plt.colorbar(scatter, label='Retention Probability phi(x,y)')
                plt.title(f'Epoch {epoch}: Mean phi(x,y)')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(os.path.join(f"{viz_dir}/Case_num{Case_num}_Stage{Stage_num}_num{Repeat_id}", f'phi_mean_epoch_{epoch}.png'))
                plt.close()
                if stage == 1 and cfg["bayesian_phi"]["update_in_stage1"] == True:
                    plt.figure(figsize=(6, 6))
                    vmin = float(mean_phi_1.min().item())
                    vmax = float(mean_phi_1.max().item())
                    scatter = plt.scatter(x, y, c=mean_phi_1_np, cmap='viridis', vmin=vmin, vmax=vmax, s=10)
                    plt.colorbar(scatter, label='Retention Probability phi(x,y)')
                    plt.title(f'Epoch {epoch}: Mean phi of Spatial part (x,y)')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.savefig(os.path.join(f"{viz_dir}/Case_num{Case_num}_Stage{Stage_num}_num{Repeat_id}", f'phi_1_epoch_{epoch}.png'))
                    plt.close()

                    plt.figure(figsize=(6, 6))
                    vmin = float(mean_phi_2.min().item())
                    vmax = float(mean_phi_2.max().item())
                    scatter = plt.scatter(x, y, c=mean_phi_2_np, cmap='viridis', vmin=vmin, vmax=vmax, s=10)
                    plt.colorbar(scatter, label='Retention Probability phi(x,y)')
                    plt.title(f'Epoch {epoch}: Mean phi of Temporal part(x,y)')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.savefig(os.path.join(f"{viz_dir}/Case_num{Case_num}_Stage{Stage_num}_num{Repeat_id}", f'phi_2_epoch_{epoch}.png'))
                    plt.close()

                # Visualization of reconstruction uncertainty map for first case in first batch of test set
                if stage == 0 and first_batch_viz_data is not None:
                    viz_Y, viz_variance = first_batch_viz_data
                    viz_x = viz_Y[:, 0]
                    viz_y = viz_Y[:, 1]

                    print(f'viz_x.shape is {viz_x.shape}')
                    print(f'viz_y.shape is {viz_y.shape}')
                    print(f'viz_variance.shape is {viz_variance.shape}')

                    # Plot scatter with color by variance
                    plt.figure(figsize=(6, 6))
                    var_vmin = np.min(viz_variance)
                    var_vmax = np.max(viz_variance)
                    scatter_var = plt.scatter(viz_x, viz_y, c=viz_variance, cmap='inferno', vmin=var_vmin, vmax=var_vmax, s=10)
                    plt.colorbar(scatter_var, label='Reconstruction Uncertainty (Variance)')
                    plt.title(f'Epoch {epoch}: Uncertainty Map (First Test Case)')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.savefig(os.path.join(f"{viz_dir}/Case_num{Case_num}_Stage{Stage_num}_num{Repeat_id}", f'uncertainty_epoch_{epoch}.png'))
                    plt.close()
                    first_batch_viz_data = None

                if stage == 0:
                    psd_out_viz = compute_psd(out[0:1], Y[0])  # [1, n_bins] for first case
                    psd_f_viz = compute_psd(G_f[0:1], Y[0])    # [1, n_bins]
                    plt.figure(figsize=(6, 4))
                    plt.loglog(range(psd_out_viz.shape[1]), psd_out_viz[0].cpu().numpy(), label='Learned PSD')
                    plt.loglog(range(psd_f_viz.shape[1]), psd_f_viz[0].cpu().numpy(), label='GT PSD')
                    plt.title(f'Epoch {epoch}: PSD Comparison')
                    plt.xlabel('Wavenumber bin')
                    plt.ylabel('Power')
                    plt.legend()
                    plt.savefig(os.path.join(f"{viz_dir}/Case_num{Case_num}_Stage{Stage_num}_num{Repeat_id}", f'psd_epoch_{epoch}.png'))
                    plt.close()
                elif stage == 1:
                    spec_out_viz = compute_spectrum(out[0:1])  # [1, T//2+1]
                    spec_f_viz = compute_spectrum(G_f[0:1])    # [1, T//2+1]
                    plt.figure(figsize=(6, 4))
                    plt.plot(range(spec_out_viz.shape[1]), spec_out_viz[0].cpu().numpy(), label='Learned Spectrum')
                    plt.plot(range(spec_f_viz.shape[1]), spec_f_viz[0].cpu().numpy(), label='GT Spectrum')
                    plt.title(f'Epoch {epoch}: Spectrum Comparison')
                    plt.xlabel('Frequency bin')
                    plt.ylabel('Power')
                    plt.legend()
                    plt.savefig(os.path.join(f"{viz_dir}/Case_num{Case_num}_Stage{Stage_num}_num{Repeat_id}", f'spectrum_epoch_{epoch}.png'))
                    plt.close()

            # early-stopping & checkpoint

            # if avg_test_loss_mse < best_test: 
            #     best_test, epochs_no_imp = avg_test_loss_mse, 0
            
            # if avg_train_loss_mse < best_test:
            #     best_test, epochs_no_imp = avg_train_loss_mse, 0

            if (avg_test_loss_mse+avg_train_loss_mse)/2 < best_test: 
                best_test, epochs_no_imp = (avg_test_loss_mse+avg_train_loss_mse)/2, 0

                ckpt_dir = pathlib.Path(cfg["save_net_dir"]); ckpt_dir.mkdir(exist_ok=True, parents=True)
                state_dict = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
                torch.save(state_dict,
                        ckpt_dir / f"Net_{Net_Name}.pth")
                print("Test loss improved, model saved.\n")

            else:
                epochs_no_imp += 1
                print(f'Not improving for {epochs_no_imp*monitor_every} epochs, the best loss is {best_test}\n')
                if epochs_no_imp >= patience_iter:
                    print(f"No improvement in {patience_iter*monitor_every} epochs. Early stopping at epoch {epoch}.")
                    return

# ----------------------------------------------------------------------
def main():
    args = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent
    project_root = script_dir.parent
    config_path = project_root / args.config

    cfg = load_cfg(config_path)

    train(cfg)

if __name__ == "__main__":

    main()

