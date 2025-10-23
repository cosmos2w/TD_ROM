
import os
import numpy as np
import matplotlib
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import gridspec
from scipy.spatial.distance import cdist

from typing import Sequence
from pathlib import Path
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_loss_history(
    epoch: int,
    train_hist: list[float],
    test_hist: list[float],
    *,
    train_hist_ch: list[list[float]] | None = None,
    test_hist_ch: list[list[float]] | None = None,
    save_dir: str | pathlib.Path = ".",
    net_num: float| None=None,
    net_name: str = "net",
    dpi: int = 150
) -> None:
    """
    Plot global and per-channel MSE histories.

    Parameters
    ----------
    epoch           : last finished epoch number (int, 1-based)
    train_hist      : length=`epoch` list of global train MSE
    test_hist       : length=`epoch` list of global test  MSE
    train_hist_ch   : optional 2-D list, shape (epoch, N_c) with per-channel train MSE
    test_hist_ch    : optional 2-D list, shape (epoch, N_c) with per-channel test  MSE
    save_dir        : directory in which to store the PNG
    net_name        : string used in the filename
    dpi             : output resolution
    """

    # --------------------------------------------------
    # preparation
    # --------------------------------------------------
    xs = list(range(1, epoch + 1))
    ys_train = [float(v) for v in train_hist]
    ys_test  = [float(v) for v in test_hist]

    # print(f'train_hist_ch is {train_hist_ch}')
    # N_c = 1 if train_hist_ch is None else len(train_hist_ch[0])

    N_c = len(train_hist_ch[0]) if train_hist_ch else 1
    n_panels = 1 if N_c == 1 else 1 + N_c     # global + each channel

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_panels,
        figsize=(4 * n_panels, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:           # make iterable
        axes = [axes]

    # --------------------------------------------------
    # panel 0 : global curves
    # --------------------------------------------------
    ax = axes[0]
    ax.semilogy(xs, ys_train, label="Global MSE Train")
    ax.semilogy(xs, ys_test,  label="Global MSE Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Global")
    ax.grid(True)
    ax.legend()

    # --------------------------------------------------
    # per-channel panels (if any)
    # --------------------------------------------------
    if N_c > 1:
        for ci in range(N_c):
            ys_ch_train = [float(row[ci]) for row in train_hist_ch]
            ys_ch_test  = [float(row[ci]) for row in test_hist_ch]

            ax = axes[ci + 1]          # offset by 1 (0 is global)
            ax.semilogy(xs, ys_ch_train, '--', label=f"Train Ch{ci}")
            ax.semilogy(xs, ys_ch_test,  '-.', label=f"Test Ch{ci}")
            ax.set_xlabel("Epoch")
            ax.set_title(f"Channel {ci}")
            ax.grid(True)
            ax.legend()

    # --------------------------------------------------
    # save & close
    # --------------------------------------------------
    # save_path = pathlib.Path(save_dir).expanduser().resolve() / f"loss_history_{net_name}.png"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # case_dir = save_path / f"Case{net_num}"
    # case_dir.mkdir(exist_ok=True, parents=True)

    base_dir = pathlib.Path(save_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    case_dir = base_dir / f"Case{net_num}"
    case_dir.mkdir(parents=True, exist_ok=True)
    figure_path = case_dir / f"loss_history_{net_name}.png"

    fig.savefig(figure_path, dpi=dpi)
    plt.close(fig)

def evaluate(u_true_recon, u_pred_phys):

    # --- Basic Input Validation ---
    if not isinstance(u_true_recon, np.ndarray) or not isinstance(u_pred_phys, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")
        
    if u_true_recon.shape != u_pred_phys.shape:
        raise ValueError(
            f"Shape mismatch: u_true_recon has shape {u_true_recon.shape} "
            f"but u_pred_phys has shape {u_pred_phys.shape}"
        )
    
    if u_true_recon.ndim != 1:
        print(f"Warning: Expected 1D arrays, but received arrays with {u_true_recon.ndim} dimensions. "
              "The array will be flattened for calculation.")
        u_true_recon = u_true_recon.flatten()
        u_pred_phys = u_pred_phys.flatten()

    # print(f'u_true_recon.shape is {u_true_recon.shape}')
    # print(f'u_pred_phys.shape is {u_pred_phys.shape}\n')

    # --- Calculations ---
    diff = u_true_recon - u_pred_phys
    
    # --- 1. Compute Mean Squared Error (MSE) ---
    global_mse = np.mean(diff ** 2)
    # print(f"Global MSE = {global_mse:.4e}")
    
    # --- 2. Compute Absolute L2 Norm: ||error||_2 ---
    # np.linalg.norm is the standard and most efficient way to compute the L2 norm.
    # It is equivalent to np.sqrt(np.sum(diff ** 2)).
    global_l2_abs = np.linalg.norm(diff)
    # print(f"Global Absolute L2 Norm = {global_l2_abs:.4e}")
    
    # --- 3. Compute Relative L2 Norm: ||error||_2 / ||true||_2 ---
    # First, compute the L2 norm of the true signal vector.
    global_true_norm = np.linalg.norm(u_true_recon)
    epsilon = 1e-10
    global_l2_rel = global_l2_abs / (global_true_norm + epsilon)
    # print(f"Global Relative L2 Norm = {global_l2_rel:.4e}")
    
    return global_mse, global_l2_rel

def _save_plot(
    u_true: np.ndarray,          # (Nt, Nx, Ny) *or* (Nt, Npts)
    u_pred: np.ndarray,          # same shape as u_true
    X: np.ndarray,               # (Npts, 2)   – columns: x, y
    times: Sequence[float],      # (Nt,)
    timesteps: Sequence[int],    # e.g. (0, 50, 75)
    out_dir: str | os.PathLike,  # directory (created if missing)
    *,
    sensor_coords: np.ndarray | None = None,
    cmap_field: str = "viridis",
    cmap_err: str = "inferno",
    dpi: int = 300,
    N_window: int = 1,
) -> None:
    """
    For each index in `timesteps` create a PNG with three panels:
        Ground-truth | Prediction | |Error|.
    Works with structured or unstructured 2-D point clouds.

    Parameters
    ----------
    u_true / u_pred
        Field values. Either (Nt, Nx, Ny) **or** already flattened
        (Nt, Npts).  The function flattens automatically if needed.
    X
        Array with the point coordinates (x, y) for all spatial locations.
        Shape (Npts, 2).  The ordering must match the *flattened* field
        ordering (row-major if the tensors were (Nx, Ny)).
    times
        Physical time stamps, one entry per stored frame.
    timesteps
        Integer indices of the frames you want to export.
    out_dir
        Output directory for the generated PNGs.
    """
    # ------------- reshape / sanity checks ---------------------------------
    u_true = np.asarray(u_true)
    u_pred = np.asarray(u_pred)
    X      = np.asarray(X)
    times  = np.asarray(times)

    if u_true.shape != u_pred.shape:
        raise ValueError("u_true and u_pred must have identical shape.")

    # Accept both (Nt, Nx, Ny) and (Nt, Npts)
    if u_true.ndim == 3:
        Nt, Nx, Ny = u_true.shape
        u_true = u_true.reshape(Nt, -1)
        u_pred = u_pred.reshape(Nt, -1)
    elif u_true.ndim == 2:
        Nt, Npts = u_true.shape
    else:
        raise ValueError("u_true/u_pred must have 2 or 3 dimensions.")

    if X.shape[0] != u_true.shape[1]:
        raise ValueError(
            "Number of points in X does not match flattened field size."
        )
    if len(times) != Nt:
        raise ValueError("`times` must have length Nt.")

    timesteps = list(timesteps)
    if not all(0 <= k < Nt for k in timesteps):
        raise ValueError(f"All timesteps must be between 0 and {Nt-1}.")

    # ------------- triangulation (once) ------------------------------------
    u_true = np.ma.masked_invalid(u_true)
    u_pred = np.ma.masked_invalid(u_pred)

    x = X[:, 0]
    y = X[:, 1]
    triang = mtri.Triangulation(x, y)

    # Mask every triangle that touches at least one NaN vertex *in u_pred*
    bad_vertices   = ~np.isfinite(u_pred[0])              # (Npts,)
    tri_mask       = ~np.all(~bad_vertices[triang.triangles], axis=1)
    triang.set_mask(tri_mask)

    # ------------- colour limits (shared across all frames) -----------------
    field_min = np.minimum(u_true.min(), u_pred.min())
    field_max = np.maximum(u_true.max(), u_pred.max())

    err_all   = np.abs(u_true - u_pred)
    err_min   = err_all[err_all > 0].min() if np.any(err_all > 0) else 0.0
    err_max   = err_all.max()

    # err_min   = 0.0
    # err_max   = 0.50

    # ------------- export directory ----------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------- iterate over requested frames ---------------------------
    PLOT_SENSORS = True
    for t_idx in timesteps:
        u_t = u_true[t_idx]          # (Npts,)
        u_p = u_pred[t_idx]
        err = np.abs(u_t - u_p)
        # mse = np.mean((u_t - u_p) ** 2)
        mse, L2Norm = evaluate(u_t, u_p)
        t_val = times[t_idx]

        if t_idx > N_window: PLOT_SENSORS = False

        fig = plt.figure(figsize=(12, 4))
        gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1],
                                wspace=0.20, hspace=0.0)

        ax_true = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_err  = fig.add_subplot(gs[0, 2])

        im_true = ax_true.tricontourf(
            triang, u_t, levels=100, cmap=cmap_field,
            vmin=field_min, vmax=field_max
        )
        im_pred = ax_pred.tricontourf(
            triang, u_p, levels=100, cmap=cmap_field,
            vmin=field_min, vmax=field_max
        )

        im_err = ax_err.tricontourf(
            triang, err, levels=100, cmap=cmap_err,
            vmin=err_min, vmax=err_max
        )

        # im_err = ax_err.tricontourf(
        #     triang, err, levels=100, cmap=cmap_err,
        #     norm=LogNorm(vmin=err_min, vmax=err_max)
        # )

        # ------------------- mark sensors ------------------------------------
        if sensor_coords is not None and PLOT_SENSORS is True:
            ax_true.scatter(
                sensor_coords[:, 0], sensor_coords[:, 1],
                s=6, c="none", edgecolors="green", linewidths=0.6,
                marker="o", zorder=4, label="sensors"
            )
            ax_true.legend(frameon=False, loc="upper right", fontsize=8)
        # ---------------------------------------------------------------------

        # ---------- cosmetics ----------
        ax_true.set_title("Ground truth")
        ax_pred.set_title("Prediction")
        ax_err.set_title("|Error|")

        for ax in (ax_true, ax_pred, ax_err):
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

        # shared colourbar for the physical field
        cbar_field = fig.colorbar(
            im_true, ax=[ax_true, ax_pred], shrink=0.6, pad=0.02
        )
        cbar_field.set_label("u")

        # colourbar for the error
        cbar_err = fig.colorbar(im_err, ax=ax_err, shrink=0.6, pad=0.02)
        cbar_err.set_label("|u - û|")

        fig.suptitle(f"t = {t_val:.3f}    |    MSE = {mse:.3e}    |     L2Norm = {L2Norm:.3e}", y=0.96)

        # ---------- save ----------
        filename = out_dir / f"frame_{t_idx:04d}.png"
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {filename}")

# Temporarily modified for channel flow
def save_plot(
    u_true: np.ndarray,          # (Nt, Nx, Ny) *or* (Nt, Npts)
    u_pred: np.ndarray,          # same shape as u_true
    X: np.ndarray,               # (Npts, 2)   – columns: x, y
    times: Sequence[float],      # (Nt,)
    timesteps: Sequence[int],    # e.g. (0, 50, 75)
    out_dir: str | os.PathLike,  # directory (created if missing)
    *,
    sensor_coords: np.ndarray | None = None,
    cmap_field: str = "viridis",
    cmap_err: str = "inferno",
    dpi: int = 300,
    N_window: int = 1,
) -> None:
    """
    For each index in `timesteps` create a PNG with three panels:
        Ground-truth | Prediction | |Error|.
    Works with structured or unstructured 2-D point clouds.

    Parameters
    ----------
    u_true / u_pred
        Field values. Either (Nt, Nx, Ny) **or** already flattened
        (Nt, Npts).  The function flattens automatically if needed.
    X
        Array with the point coordinates (x, y) for all spatial locations.
        Shape (Npts, 2).  The ordering must match the *flattened* field
        ordering (row-major if the tensors were (Nx, Ny)).
    times
        Physical time stamps, one entry per stored frame.
    timesteps
        Integer indices of the frames you want to export.
    out_dir
        Output directory for the generated PNGs.
    """
    # ------------- reshape / sanity checks ---------------------------------
    u_true = np.asarray(u_true)
    u_pred = np.asarray(u_pred)
    X      = np.asarray(X)
    times  = np.asarray(times)

    if u_true.shape != u_pred.shape:
        raise ValueError("u_true and u_pred must have identical shape.")

    # Accept both (Nt, Nx, Ny) and (Nt, Npts)
    if u_true.ndim == 3:
        Nt, Nx, Ny = u_true.shape
        u_true = u_true.reshape(Nt, -1)
        u_pred = u_pred.reshape(Nt, -1)
    elif u_true.ndim == 2:
        Nt, Npts = u_true.shape
    else:
        raise ValueError("u_true/u_pred must have 2 or 3 dimensions.")

    if X.shape[0] != u_true.shape[1]:
        raise ValueError(
            "Number of points in X does not match flattened field size."
        )
    if len(times) != Nt:
        raise ValueError("`times` must have length Nt.")

    timesteps = list(timesteps)
    if not all(0 <= k < Nt for k in timesteps):
        raise ValueError(f"All timesteps must be between 0 and {Nt-1}.")

    # ------------- triangulation (once) ------------------------------------
    u_true = np.ma.masked_invalid(u_true)
    u_pred = np.ma.masked_invalid(u_pred)

    x = X[:, 0]
    y = X[:, 1]
    triang = mtri.Triangulation(x, y)

    # Mask every triangle that touches at least one NaN vertex *in u_pred*
    bad_vertices   = ~np.isfinite(u_pred[0])              # (Npts,)
    tri_mask       = ~np.all(~bad_vertices[triang.triangles], axis=1)
    triang.set_mask(tri_mask)

    # ------------- colour limits (shared across all frames) -----------------
    field_min = np.minimum(u_true.min(), u_pred.min())
    field_max = np.maximum(u_true.max(), u_pred.max())

    err_all   = np.abs(u_true - u_pred)
    err_min   = err_all[err_all > 0].min() if np.any(err_all > 0) else 0.0
    err_max   = err_all.max()

    # err_min   = 0.0
    # err_max   = 0.50

    # ------------- export directory ----------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------- iterate over requested frames ---------------------------
    PLOT_SENSORS = True
    for t_idx in timesteps:
        u_t = u_true[t_idx]          # (Npts,)
        u_p = u_pred[t_idx]
        err = np.abs(u_t - u_p)
        # mse = np.mean((u_t - u_p) ** 2)
        mse, L2Norm = evaluate(u_t, u_p)
        t_val = times[t_idx]

        if t_idx > N_window: PLOT_SENSORS = False

        # fig = plt.figure(figsize=(12, 4))
        # gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1],
        #                         wspace=0.20, hspace=0.0)
        # ax_true = fig.add_subplot(gs[0, 0])
        # ax_pred = fig.add_subplot(gs[0, 1])
        # ax_err  = fig.add_subplot(gs[0, 2])

        fig = plt.figure(figsize=(12, 8))
        gs  = gridspec.GridSpec(3, 1,
                                wspace=0.0, hspace=0.02)
        ax_true = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[1, 0])
        ax_err  = fig.add_subplot(gs[2, 0])

        im_true = ax_true.tricontourf(
            triang, u_t, levels=100, cmap=cmap_field,
            vmin=field_min, vmax=field_max
        )
        im_pred = ax_pred.tricontourf(
            triang, u_p, levels=100, cmap=cmap_field,
            vmin=field_min, vmax=field_max
        )

        im_err = ax_err.tricontourf(
            triang, err, levels=100, cmap=cmap_err,
            vmin=err_min, vmax=err_max
        )

        # ------------------- mark sensors ------------------------------------
        if sensor_coords is not None and PLOT_SENSORS is True:
            ax_true.scatter(
                sensor_coords[:, 0], sensor_coords[:, 1],
                s=6, c="none", edgecolors="green", linewidths=0.6,
                marker="o", zorder=4, label="sensors"
            )
            ax_true.legend(frameon=False, loc="upper right", fontsize=8)
        # ---------------------------------------------------------------------

        # ---------- cosmetics ----------
        ax_true.set_title("Ground truth")
        ax_pred.set_title("Prediction")
        ax_err.set_title("|Error|")

        for ax in (ax_true, ax_pred, ax_err):
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_xticks([]) 
            ax.set_yticks([]) 

        # shared colourbar for the physical field
        cbar_field = fig.colorbar(
            im_true, ax=[ax_true, ax_pred], shrink=0.6, pad=0.02
        )
        cbar_field.set_label("u")

        # colourbar for the error
        cbar_err = fig.colorbar(im_err, ax=ax_err, shrink=0.6, pad=0.02)
        cbar_err.set_label("|u - û|")

        fig.suptitle(f"t = {t_val:.3f}    |    MSE = {mse:.3e}    |     L2Norm = {L2Norm:.3e}", y=0.96)

        # ---------- save ----------
        filename = out_dir / f"frame_{t_idx:04d}.png"
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {filename}")



