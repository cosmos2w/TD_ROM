
from __future__ import annotations
from torch import nn
import argparse, datetime, pathlib, shutil, yaml, torch, csv

from src.dataloading import make_loaders
from src.models      import TDROMEncoder_Sen, Decoder_Sen, TemporalDecoderSoftmax, TD_ROM
from src.utils.plot_utils import plot_loss_history

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="collinear_flow_Re100",
        type=str,
        help="Datasets: channel_flow, collinear_flow_Re40, collinear_flow_Re100, cylinder_flow, FN_reaction_diffusion, sea_temperature, turbulent_combustion",
    )
    p.add_argument(
        "--config",
        type=pathlib.Path,
        help="YAML config file (defaults to Save_config_files/<dataset>/YAML_config_Senseiver.yaml)",
    )
    args = p.parse_args()

    if args.config is None:
        args.config = pathlib.Path(f"Save_config_files/{args.dataset}/YAML_config_Senseiver.yaml")

    return args

# ---------------------------------------------------------
def load_cfg(yaml_path: pathlib.Path) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    # ---------- snapshot ----------
    ts  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    idx = cfg.get("case_index", "X")
    st  = cfg.get("Stage", "X")
    num  = cfg.get("Repeat_id", "X")  
    snap = yaml_path.parent / "config_bk_Senseiver" / f"config_Senseiver_idx_{idx}_st_{st}_num_{num}_{ts}.yaml"
    shutil.copyfile(yaml_path, snap)
    print(f"Now copied config → {snap}")
    return cfg

# ---------------------------------------------------------
def build_model(cfg: dict,  N_c: int, device: torch.device) -> TD_ROM:

    Net_Name = f"Senseiver_id{cfg['case_index']}_st{cfg['Stage']}_num{cfg['Repeat_id']}"

    encoder = TDROMEncoder_Sen(input_ch         = N_c,
                               coord_dim        = cfg["Spatial_Dim"],
                               domain_sizes     = [ cfg["Num_x"], cfg["Num_y"] ],
                               preproc_ch          = cfg["F_dim"],
                               num_latents         = cfg["latent_tokens"],
                               num_latent_channels = cfg["F_dim"],
                               num_layers          = cfg["num_layers"],
                               num_cross_attention_heads = cfg["num_heads"],
                               num_self_attention_heads  = cfg["num_heads"],
                               num_self_attention_layers_per_block = cfg["num_layers_block"],
                               latent_to_feat_dim                  = cfg["F_dim"],
                               
                               num_pos_bands                       = 64
                            )      
    field_dec = Decoder_Sen(coord_dim          = cfg["Spatial_Dim"],
                            domain_sizes     = [ cfg["Num_x"], cfg["Num_y"] ],
                                preproc_ch          = cfg["F_dim"],
                                num_latent_channels = cfg["F_dim"],
                                latent_size         = cfg["latent_size"],
                                num_cross_attention_heads = cfg["num_heads"],
                                num_output_channels = N_c)

    decoder_lat = TemporalDecoderSoftmax(
        d_model = cfg["F_dim"],
        n_layers= cfg["num_layers_propagator"],
        n_heads = cfg["num_heads"],
        dt      = cfg["delta_t"],
    )

    net = TD_ROM(encoder, decoder_lat, field_dec, 
                delta_t = cfg["delta_t"], N_window = cfg["N_window"], stage=cfg["Stage"]) 

    return net, Net_Name

# ---------------------------------------------------------------
def train(cfg: dict):

    device_ids = cfg["device_ids"]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    # 1) data --------------------------------------------------
    train_ld, test_ld, N_c, Num_all_recon_pts = make_loaders(
        cfg["data_h5"],
        num_time_sample = cfg["num_time_sample"],
        num_space_sample= cfg["num_space_sample"],
        multi_factor    = cfg["multi_factor"],
        train_ratio     = cfg["train_ratio"],
        batch_size      = cfg["batch_size"],
        workers         = cfg["num_workers"],
        channel         = cfg["channel"],
        process_mode    = cfg["process_mode"],
        num_samples     = cfg["num_samples"],
        Full_Field_DownS= cfg["Full_Field_DownS"],
        global_restriction = cfg["global_restriction"],
        sample_restriction = cfg["sample_restriction"],
        sample_params      = cfg["Sample_Parameters"],
    )

    # 2) model / opt ------------------------------------------
    stage=cfg["Stage"]
    model, Net_Name = build_model(cfg, N_c, device)

    # Freeze parameters based on stage
    if stage == 0:  
        for param in model.temporaldecoder.parameters():
            param.requires_grad = False

    elif stage == 1:
        for p in model.fieldencoder.parameters():        p.requires_grad = False
        for p in model.decoder.parameters():             p.requires_grad = False


    # --- multi-GPU ---
    if len(device_ids) > 1:
        print(f"Using DataParallel on GPUs {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids) 
    model = model.to(device)        # send (the wrapper’s) parameters/shards to GPUs

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    # Warm-up scheduler
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

    Case_num = cfg["case_index"]
    loss_csv  = pathlib.Path(cfg["save_loss_dir"])
    loss_csv.mkdir(exist_ok=True, parents=True)
    case_dir = loss_csv / f"Case{Case_num}"
    case_dir.mkdir(exist_ok=True, parents=True)

    csv_path  = case_dir / f"loss_log_{Net_Name}.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([f"created {datetime.datetime.now()}"])
        header = ['epoch', 'train_loss', 'train_loss_sum_mse']
        if N_c > 1:
            for ci in range(N_c):
                header.append(f'train_loss_mse_ch{ci}')
        header.extend(['train_loss_TrajInfer', 'test_loss', 'test_loss_sum_mse'])
        if N_c > 1:
            for ci in range(N_c):
                header.append(f'test_loss_mse_ch{ci}')
        header.extend(['test_loss_TrajInfer'])
        csv.writer(f).writerow(header)

    best_test, epochs_no_imp = float("inf"), 0
    monitor_every = cfg["monitor_every"]
    patience_iter = cfg["patience_epochs"] // monitor_every

    train_hist, test_hist, train_hist_ch, test_hist_ch = [], [], [], []

    for epoch in range(1, cfg["num_epochs"] + 1):

        teacher_force_p = max(0.05, 0.50 - epoch / 1000)
        w_traj = 1.0

        # ----------------- TRAIN -----------------
        model.train()
        train_loss_list     = []
        train_loss_mse_list = []
        train_loss_Traj_list = []
        if N_c > 1:
            train_loss_mse_ch_list = [ [] for _ in range(N_c) ]

        for G_d, G_dt, G_f, Y, U in train_ld:
            G_d, G_dt, G_f, Y, U = map(lambda x: x.to(device), (G_d, G_dt, G_f, Y, U))
            B, num_time_sample, N_x, N_call = G_d.shape  

            opt.zero_grad()
            out, out_logvar, obs, traj, traj_logvar, G_u_cls, G_u_mean_Sens,G_u_logvar_Sens = model(G_d, G_f, Y, U, teacher_force_p)

            if N_c > 1: # Vectorized per-channel MSE
                loss_mse_channels = [mse(out[..., ci], G_f[..., ci]) for ci in range(N_c)]
                loss_mse  = sum(loss_mse_channels)
            else:
                loss_mse  = mse(out, G_f)
            
            if cfg["N_window"] == num_time_sample or stage == 0: loss_traj = 0.0
            else:
                loss_traj = w_traj * mse(traj[:, cfg["N_window"]:num_time_sample, :], obs[:, cfg["N_window"]:, :])

            loss = loss_mse + loss_traj

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss_list.append(loss )
            train_loss_mse_list.append(loss_mse )
            train_loss_Traj_list.append(loss_traj )
            if N_c > 1:
                for ci, l in enumerate(loss_mse_channels):
                    train_loss_mse_ch_list[ci].append(l)

        avg_train_loss          = sum(train_loss_list)      / len(train_loss_list)
        avg_train_loss_mse      = sum(train_loss_mse_list)  / len(train_loss_mse_list)
        if N_c > 1: avg_train_loss_mse_ch = [ sum(l)/len(l) for l in train_loss_mse_ch_list ]
        avg_train_loss_Traj     = sum(train_loss_Traj_list) / len(train_loss_Traj_list)

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
            test_loss_Traj_list = []
            if N_c > 1:
                test_loss_mse_ch_list = [ [] for _ in range(N_c) ]

            for G_d, G_dt, G_f, Y, U in test_ld:

                G_d, G_dt, G_f, Y, U = map(lambda x: x.to(device), (G_d, G_dt, G_f, Y, U))
                B, num_time_sample, N_x, N_call = G_d.shape

                out, out_logvar, obs, traj, traj_logvar, G_u_cls, G_u_mean_Sens,G_u_logvar_Sens = model(G_d, G_f, Y, U, teacher_force_p)

                if N_c > 1: # Vectorized per-channel MSE
                    loss_mse_channels = [mse(out[..., ci], G_f[..., ci]) for ci in range(N_c)]
                    loss_mse  = sum(loss_mse_channels)
                else:
                    loss_mse  = mse(out, G_f)

                if cfg["N_window"] == num_time_sample or stage == 0: loss_traj = 0.0
                else:
                    loss_traj = w_traj * mse(traj[:, cfg["N_window"]:num_time_sample, :], obs[:, cfg["N_window"]:, :])

                loss = loss_mse + loss_traj

                test_loss_list.append(loss )
                test_loss_mse_list.append(loss_mse )
                test_loss_Traj_list.append(loss_traj )
                if N_c > 1:
                    for ci, l in enumerate(loss_mse_channels):
                        test_loss_mse_ch_list[ci].append(l)

            avg_test_loss       = sum(test_loss_list)       / len(test_loss_list)
            avg_test_loss_mse   = sum(test_loss_mse_list)   / len(test_loss_mse_list)
            avg_test_loss_Traj  = sum(test_loss_Traj_list)  / len(test_loss_Traj_list)
            if N_c > 1:
                avg_test_loss_mse_ch   = [ sum(l)/len(l) for l in test_loss_mse_ch_list ]

        train_hist.append(avg_train_loss_mse.item())
        test_hist.append(avg_test_loss_mse.item())
        if N_c > 1:
            train_hist_ch.append(avg_train_loss_mse_ch)
            test_hist_ch.append(avg_test_loss_mse_ch)

        # ----------------- logging -----------------
        if epoch % monitor_every == 0:

            # build per‐channel summaries
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
                f" Train Loss: {avg_train_loss:.6f} (MSE: {train_mse_summary}, U_Traj: {avg_train_loss_Traj:.6f})\n"
                f" Test  Loss: {avg_test_loss:.6f} (MSE: {test_mse_summary}, U_Traj: {avg_test_loss_Traj:.6f})\n "
            )

            with open(csv_path, "a", newline="") as f:
                row = [epoch, f"{avg_train_loss:.6f}", f"{avg_train_loss_mse:.6f}"]
                if N_c > 1:
                    for loss_ch in avg_train_loss_mse_ch:
                        row.append(f"{loss_ch:.6f}")
                row.extend([f"{avg_train_loss_Traj:.6f}",
                            f"{avg_test_loss:.6f}", f"{avg_test_loss_mse:.6f}"])
                if N_c > 1:
                    for loss_ch in avg_test_loss_mse_ch:
                        row.append(f"{loss_ch:.6f}")
                row.extend([f"{avg_test_loss_Traj:.6f}"])
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

            # early-stopping & checkpoint
            if avg_test_loss_mse < best_test:
                best_test, epochs_no_imp = avg_test_loss_mse, 0
                ckpt_dir = pathlib.Path(cfg["save_net_dir"]); ckpt_dir.mkdir(exist_ok=True, parents=True)
                state_dict = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
                torch.save(state_dict,
                        ckpt_dir / f"Net_{Net_Name}.pth")
                print("Test loss improved, model saved.\n")

            else:
                epochs_no_imp += 1
                print(f'Not improving for {epochs_no_imp*monitor_every} epochs, the best test loss is {best_test}\n')
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