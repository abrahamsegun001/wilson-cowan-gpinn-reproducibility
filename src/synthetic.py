"""Synthetic Wilson-Cowan controls and calibration experiments.

Each command below is a namespaced consolidation of a script from the validated
reproducibility archive. The numerical implementation is preserved while the
public repository exposes a compact command surface.
"""
from __future__ import annotations

import sys as _sys

# ============================================================================
# generate: source synthetic_identifiability_package/generate_synthetic_wc.py
# ============================================================================
def generate(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['synthetic_identifiability_package/generate_synthetic_wc.py'] + list(argv or [])
    try:
        """Generate independent synthetic Wilson-Cowan data with finite differences + RK4.

        This forward solver is intentionally independent of the PINN. It solves
            tau * E_t = -E + sigmoid(w_EE E + D_E Laplacian(E) + h)
        on a 10 mm x 10 mm square using a 5-point finite-difference Laplacian,
        zero-normal-flux (Neumann) boundaries, and RK4 time stepping.
        """
        import argparse
        import json
        from pathlib import Path

        import numpy as np


        def sigmoid(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-x))


        def laplacian_neumann(E: np.ndarray, dx: float) -> np.ndarray:
            # Edge padding implements zero normal derivative to first order.
            p = np.pad(E, 1, mode="edge")
            return (
                p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * E
            ) / (dx * dx)


        def rhs(E: np.ndarray, D_E: float, w_EE: float, h: float, tau: float, dx: float) -> np.ndarray:
            lap = laplacian_neumann(E, dx)
            return (-E + sigmoid(w_EE * E + D_E * lap + h)) / tau


        def make_initial_condition(nx: int, ny: int, l_fov: float) -> np.ndarray:
            x = np.linspace(-l_fov / 2.0, l_fov / 2.0, nx, dtype=np.float64)
            y = np.linspace(-l_fov / 2.0, l_fov / 2.0, ny, dtype=np.float64)
            yy, xx = np.meshgrid(y, x, indexing="ij")
            r = np.sqrt(xx**2 + yy**2)

            # Smooth high-activity domain embedded in a low-activity background.
            # It creates nonzero Laplacian information without injecting a time-dependent forcing.
            E0 = 0.05 + 0.90 / (1.0 + np.exp((r - 1.8) / 0.18))
            # Mild asymmetric perturbation prevents a perfectly radial inverse problem.
            E0 += 0.035 * np.exp(-((xx + 2.5) ** 2 + (yy - 2.0) ** 2) / (2.0 * 0.55**2))
            return np.clip(E0, 0.01, 0.99)


        def simulate(
            D_E: float,
            w_EE: float,
            h: float,
            tau: float,
            l_fov: float,
            nx: int,
            ny: int,
            duration: float,
            output_hz: float,
            substeps_per_frame: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            if nx != ny:
                raise ValueError("This generator currently assumes nx == ny so dx == dy.")
            dx = l_fov / (nx - 1)
            dt_frame = 1.0 / output_hz
            dt = dt_frame / substeps_per_frame
            n_frames = int(round(duration * output_hz))

            E = make_initial_condition(nx, ny, l_fov)
            frames = np.empty((n_frames, ny, nx), dtype=np.float32)

            for frame in range(n_frames):
                for _ in range(substeps_per_frame):
                    k1 = rhs(E, D_E, w_EE, h, tau, dx)
                    k2 = rhs(E + 0.5 * dt * k1, D_E, w_EE, h, tau, dx)
                    k3 = rhs(E + 0.5 * dt * k2, D_E, w_EE, h, tau, dx)
                    k4 = rhs(E + dt * k3, D_E, w_EE, h, tau, dx)
                    E = E + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                    E = np.clip(E, 0.0, 1.0)
                frames[frame] = E.astype(np.float32)

            t = np.arange(n_frames, dtype=np.float64) / output_hz
            return frames, t


        def main() -> None:
            p = argparse.ArgumentParser()
            p.add_argument("--output", type=str, default="synthetic_data/wc_twin_truth.npz")
            p.add_argument("--D_E", type=float, default=0.60)
            p.add_argument("--w_EE", type=float, default=5.20)
            p.add_argument("--h", type=float, default=-2.60)
            p.add_argument("--tau", type=float, default=1.0)
            p.add_argument("--l_fov", type=float, default=10.0)
            p.add_argument("--nx", type=int, default=128)
            p.add_argument("--ny", type=int, default=128)
            p.add_argument("--duration", type=float, default=30.0)
            p.add_argument("--output_hz", type=float, default=30.0)
            p.add_argument("--substeps_per_frame", type=int, default=4)
            args = p.parse_args()

            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)

            E, t = simulate(
                args.D_E, args.w_EE, args.h, args.tau, args.l_fov,
                args.nx, args.ny, args.duration, args.output_hz, args.substeps_per_frame,
            )

            metadata = {
                "forward_solver": "5-point finite-difference Laplacian + explicit RK4",
                "boundary_condition": "homogeneous Neumann (zero normal flux)",
                "D_E_true": float(args.D_E),
                "w_EE_true": float(args.w_EE),
                "h_true": float(args.h),
                "tau": float(args.tau),
                "l_fov_mm": float(args.l_fov),
                "nx": int(args.nx),
                "ny": int(args.ny),
                "duration_s": float(args.duration),
                "output_hz": float(args.output_hz),
                "substeps_per_frame": int(args.substeps_per_frame),
                "integration_dt_s": float(1.0 / args.output_hz / args.substeps_per_frame),
                "normalization": "none; E is generated directly on the model scale [0,1]",
            }

            np.savez_compressed(
                out,
                E=E,
                t_seconds=t,
                D_E_true=np.float64(args.D_E),
                w_EE_true=np.float64(args.w_EE),
                h_true=np.float64(args.h),
                tau=np.float64(args.tau),
                l_fov=np.float64(args.l_fov),
                metadata_json=np.array(json.dumps(metadata)),
            )
            print(json.dumps(metadata, indent=2))
            print(f"Saved: {out}")
            print(f"E shape={E.shape}; min={E.min():.6f}, max={E.max():.6f}, std={E.std():.6f}")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# baseline_inverse: source synthetic_identifiability_package/train_synthetic_gpinn.py
# ============================================================================
def baseline_inverse(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['synthetic_identifiability_package/train_synthetic_gpinn.py'] + list(argv or [])
    try:
        """Single-seed inverse recovery on independently generated Wilson-Cowan data."""
        import argparse
        import json
        from pathlib import Path

        import numpy as np
        import torch

        from .model import (
            gPINN_WilsonCowan,
            set_seed,
            train_single_chunk,
        )


        def flat_indices_to_xyt(indices: np.ndarray, T: int, H: int, W: int) -> np.ndarray:
            # Flattening order corresponds to E[t, y, x].
            hw = H * W
            ti = indices // hw
            rem = indices % hw
            yi = rem // W
            xi = rem % W

            x = -1.0 + 2.0 * xi.astype(np.float32) / float(W - 1)
            y = -1.0 + 2.0 * yi.astype(np.float32) / float(H - 1)
            t = -1.0 + 2.0 * ti.astype(np.float32) / float(T - 1)
            return np.stack([x, y, t], axis=1).astype(np.float32)


        def make_sampled_tensors(
            E_clean: np.ndarray,
            n_data: int,
            n_colloc: int,
            noise_sigma: float,
            noise_seed: int,
            sampling_seed: int,
            device: torch.device,
        ):
            T, H, W = E_clean.shape
            n_total = T * H * W
            n_train = min(int(n_data), int(0.8 * n_total))
            n_val = min(int(n_data * 0.25), n_total - n_train)

            # A fixed sampling seed means all training seeds see exactly the same
            # supervised and collocation locations, isolating network/optimization seed.
            rng = np.random.default_rng(int(sampling_seed))
            chosen = rng.choice(n_total, size=n_train + n_val, replace=False)
            idx_tr = chosen[:n_train]
            idx_va = chosen[n_train:]
            idx_c = rng.choice(n_total, size=min(n_colloc, n_total), replace=False)

            flat_clean = E_clean.reshape(-1).astype(np.float32)
            if noise_sigma > 0:
                nrng = np.random.default_rng(int(noise_seed))
                noise = nrng.normal(0.0, noise_sigma, size=flat_clean.shape).astype(np.float32)
                flat_obs = np.clip(flat_clean + noise, 0.0, 1.0)
            else:
                flat_obs = flat_clean

            Xt_tr_np = flat_indices_to_xyt(idx_tr, T, H, W)
            Xt_va_np = flat_indices_to_xyt(idx_va, T, H, W)
            Xc_np = flat_indices_to_xyt(idx_c, T, H, W)

            Xt_tr = torch.from_numpy(Xt_tr_np).to(device)
            Et_tr = torch.from_numpy(flat_obs[idx_tr, None]).to(device)
            Xt_va = torch.from_numpy(Xt_va_np).to(device)
            # Validation is against the same observed/noisy target, matching the real pipeline.
            Et_va = torch.from_numpy(flat_obs[idx_va, None]).to(device)
            # Also retain clean validation target for a noise-independent reconstruction metric.
            Et_va_clean = torch.from_numpy(flat_clean[idx_va, None]).to(device)
            Xc = torch.from_numpy(Xc_np).to(device).requires_grad_(True)
            return Xt_tr, Et_tr, Xt_va, Et_va, Et_va_clean, Xc


        def metrics_against_clean(model, Xt, Et_clean):
            model.eval()
            with torch.no_grad():
                pred = model.forward_E(Xt)
                mse = torch.mean((pred - Et_clean) ** 2).item()
                true_np = Et_clean.cpu().numpy().ravel()
                pred_np = pred.cpu().numpy().ravel()
            ss_res = float(np.sum((true_np - pred_np) ** 2))
            ss_tot = float(np.sum((true_np - true_np.mean()) ** 2))
            r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
            model.train()
            return float(mse), r2


        def safe_tag(x: float) -> str:
            return f"{x:.3f}".replace("-", "m").replace(".", "p")


        def main() -> None:
            p = argparse.ArgumentParser()
            p.add_argument("--data", type=str, default="synthetic_data/wc_twin_truth.npz")
            p.add_argument("--output_dir", type=str, default="synthetic_outputs")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--fourier_seed", type=int, default=2718)
            p.add_argument("--sampling_seed", type=int, default=31415)
            p.add_argument("--noise_seed", type=int, default=1618)
            p.add_argument("--noise_sigma", type=float, default=0.0)
            p.add_argument("--n_data", type=int, default=60000)
            p.add_argument("--n_colloc", type=int, default=90000)
            p.add_argument("--adam_epochs", type=int, default=5000)
            p.add_argument("--lbfgs_steps", type=int, default=500)
            p.add_argument("--lambda_physics", type=float, default=0.01)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--mu_grad", type=float, default=1e-5)
            args = p.parse_args()

            set_seed(args.seed)
            torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            d = np.load(args.data, allow_pickle=False)
            E = d["E"].astype(np.float32)
            truth = {
                "D_E": float(d["D_E_true"]),
                "w_EE": float(d["w_EE_true"]),
                "h": float(d["h_true"]),
            }

            Xt_tr, Et_tr, Xt_va, Et_va, Et_va_clean, Xc = make_sampled_tensors(
                E, args.n_data, args.n_colloc, args.noise_sigma,
                args.noise_seed, args.sampling_seed, device,
            )

            # Fixed Fourier seed + fixed sampled points across replicate training seeds.
            # Therefore replicate variation is attributable to trainable-network initialization
            # and optimization, not a different synthetic dataset or a different point split.
            model = gPINN_WilsonCowan(fourier_seed=args.fourier_seed).to(device)
            result = train_single_chunk(
                model, Xt_tr, Et_tr, Xt_va, Et_va, Xc,
                lr=args.lr,
                adam_epochs=args.adam_epochs,
                lbfgs_steps=args.lbfgs_steps,
                lambda_phys=args.lambda_physics,
                mu_grad=args.mu_grad,
            )

            clean_mse, clean_r2 = metrics_against_clean(model, Xt_va, Et_va_clean)
            est = {
                "D_E": float(model.forward_D().item()),
                "w_EE": float(model.w_EE.item()),
                "h": float(model.h.item()),
                "raw_D": float(model.raw_D.item()),
            }

            rel_error = {
                k: abs(est[k] - truth[k]) / max(abs(truth[k]), 1e-12) * 100.0
                for k in ("D_E", "w_EE", "h")
            }

            out = Path(args.output_dir)
            (out / "results").mkdir(parents=True, exist_ok=True)
            (out / "checkpoints").mkdir(parents=True, exist_ok=True)
            (out / "histories").mkdir(parents=True, exist_ok=True)

            tag = f"seed-{args.seed}_noise-{safe_tag(args.noise_sigma)}"
            checkpoint_path = out / "checkpoints" / f"{tag}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": args.seed,
                    "fourier_seed": args.fourier_seed,
                    "sampling_seed": args.sampling_seed,
                    "noise_seed": args.noise_seed,
                    "noise_sigma": args.noise_sigma,
                    "truth": truth,
                    "estimate": est,
                },
                checkpoint_path,
            )

            np.savez_compressed(
                out / "histories" / f"{tag}.npz",
                **{k: np.asarray(v) for k, v in result.history.items()},
            )

            summary = {
                "seed": int(args.seed),
                "fourier_seed": int(args.fourier_seed),
                "sampling_seed": int(args.sampling_seed),
                "noise_seed": int(args.noise_seed),
                "noise_sigma": float(args.noise_sigma),
                "device": str(device),
                "truth": truth,
                "estimate": est,
                "relative_error_percent": rel_error,
                "val_mse_observed": float(result.val_mse),
                "val_r2_observed": float(result.val_r2),
                "val_r_observed": float(result.val_r),
                "val_mse_clean": float(clean_mse),
                "val_r2_clean": float(clean_r2),
                "final_lambda": float(result.final_lambda),
                "checkpoint": str(checkpoint_path),
                "n_data": int(args.n_data),
                "n_colloc": int(args.n_colloc),
                "adam_epochs": int(args.adam_epochs),
                "lbfgs_steps": int(args.lbfgs_steps),
                "mu_grad": float(args.mu_grad),
            }
            result_path = out / "results" / f"{tag}.json"
            result_path.write_text(json.dumps(summary, indent=2))

            print(json.dumps(summary, indent=2))
            print(f"Saved: {result_path}")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# plot_baseline: source synthetic_identifiability_package/plot_synthetic_results.py
# ============================================================================
def plot_baseline(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['synthetic_identifiability_package/plot_synthetic_results.py'] + list(argv or [])
    try:
        """Aggregate synthetic recovery runs and create manuscript-ready diagnostics."""
        import argparse
        import csv
        import json
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        from .model import gPINN_WilsonCowan


        def load_jsons(results_dir: Path):
            rows = []
            for path in sorted(results_dir.glob("seed-*_noise-*.json")):
                with open(path) as f:
                    row = json.load(f)
                row["_path"] = str(path)
                rows.append(row)
            if not rows:
                raise FileNotFoundError(f"No result JSON files found in {results_dir}")
            return rows


        def write_summary_csv(rows, out_path: Path):
            fields = [
                "noise_sigma", "seed",
                "D_E_true", "D_E_est", "D_E_relerr_pct",
                "w_EE_true", "w_EE_est", "w_EE_relerr_pct",
                "h_true", "h_est", "h_relerr_pct",
                "raw_D", "val_r2_observed", "val_r2_clean", "val_mse_clean",
            ]
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow({
                        "noise_sigma": r["noise_sigma"], "seed": r["seed"],
                        "D_E_true": r["truth"]["D_E"], "D_E_est": r["estimate"]["D_E"],
                        "D_E_relerr_pct": r["relative_error_percent"]["D_E"],
                        "w_EE_true": r["truth"]["w_EE"], "w_EE_est": r["estimate"]["w_EE"],
                        "w_EE_relerr_pct": r["relative_error_percent"]["w_EE"],
                        "h_true": r["truth"]["h"], "h_est": r["estimate"]["h"],
                        "h_relerr_pct": r["relative_error_percent"]["h"],
                        "raw_D": r["estimate"]["raw_D"],
                        "val_r2_observed": r["val_r2_observed"],
                        "val_r2_clean": r["val_r2_clean"],
                        "val_mse_clean": r["val_mse_clean"],
                    })


        def aggregate_table(rows, out_path: Path):
            lines = []
            for noise in sorted({float(r["noise_sigma"]) for r in rows}):
                subset = [r for r in rows if float(r["noise_sigma"]) == noise]
                truth = subset[0]["truth"]
                lines.append({"noise_sigma": noise, "n": len(subset)})
                for p in ("D_E", "w_EE", "h"):
                    vals = np.array([r["estimate"][p] for r in subset], dtype=float)
                    rel = np.abs(vals - truth[p]) / max(abs(truth[p]), 1e-12) * 100.0
                    lines[-1][f"{p}_true"] = truth[p]
                    lines[-1][f"{p}_mean"] = vals.mean()
                    lines[-1][f"{p}_sd"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                    lines[-1][f"{p}_mean_relerr_pct"] = rel.mean()
                    lines[-1][f"{p}_cv_pct"] = (
                        lines[-1][f"{p}_sd"] / abs(vals.mean()) * 100.0 if abs(vals.mean()) > 1e-12 else np.nan
                    )

            keys = list(lines[0].keys())
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(lines)
            return lines


        def plot_parameter(rows, parameter: str, figure_dir: Path):
            noises = sorted({float(r["noise_sigma"]) for r in rows})
            truth = float(rows[0]["truth"][parameter])
            fig, ax = plt.subplots(figsize=(6.2, 4.5))
            for i, noise in enumerate(noises):
                subset = [r for r in rows if float(r["noise_sigma"]) == noise]
                vals = np.array([r["estimate"][parameter] for r in subset])
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(np.full(len(vals), i) + jitter, vals, label=f"σ={noise:g}")
                ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0.0,
                            fmt="o", capsize=4)
            ax.axhline(truth, linestyle="--", label="Ground truth")
            ax.set_xticks(range(len(noises)), [f"σ={n:g}" for n in noises])
            ax.set_ylabel(parameter)
            ax.set_xlabel("Observation-noise level")
            ax.set_title(f"Synthetic recovery of {parameter}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figure_dir / f"recovery_{parameter}.png", dpi=300)
            fig.savefig(figure_dir / f"recovery_{parameter}.pdf")
            plt.close(fig)


        def finite_difference_terms(E: np.ndarray, dt: float, dx: float):
            # Central time derivative; spatial derivatives use interior pixels only.
            dEdt = (E[2:] - E[:-2]) / (2.0 * dt)
            mid = E[1:-1]
            lap = (
                mid[:, 1:-1, 2:] + mid[:, 1:-1, :-2]
                + mid[:, 2:, 1:-1] + mid[:, :-2, 1:-1]
                - 4.0 * mid[:, 1:-1, 1:-1]
            ) / (dx * dx)
            return mid[:, 1:-1, 1:-1], dEdt[:, 1:-1, 1:-1], lap


        def residual_mse(E, dEdt, lap, D, w, h, tau=1.0):
            u = np.clip(w * E + D * lap + h, -40.0, 40.0)
            S = 1.0 / (1.0 + np.exp(-u))
            f = tau * dEdt - (-E + S)
            return float(np.mean(f * f))


        def compute_and_plot_landscape(data_path: Path, figure_dir: Path, max_points: int = 50000):
            d = np.load(data_path)
            Eall = d["E"].astype(np.float64)
            D0 = float(d["D_E_true"]); w0 = float(d["w_EE_true"]); h0 = float(d["h_true"])
            tau = float(d["tau"]); L = float(d["l_fov"])
            dt = 1.0 / 30.0
            dx = L / (Eall.shape[2] - 1)
            E, dEdt, lap = finite_difference_terms(Eall, dt, dx)
            E = E.ravel(); dEdt = dEdt.ravel(); lap = lap.ravel()
            rng = np.random.default_rng(2026)
            if E.size > max_points:
                idx = rng.choice(E.size, max_points, replace=False)
                E, dEdt, lap = E[idx], dEdt[idx], lap[idx]

            configs = [
                ("D_E", "w_EE", np.linspace(max(0.11, D0 * 0.35), min(4.9, D0 * 2.1), 41),
                 np.linspace(max(0.55, w0 - 1.4), min(5.95, w0 + 0.7), 41)),
                ("D_E", "h", np.linspace(max(0.11, D0 * 0.35), min(4.9, D0 * 2.1), 41),
                 np.linspace(h0 - 1.0, h0 + 1.0, 41)),
                ("w_EE", "h", np.linspace(max(0.55, w0 - 1.4), min(5.95, w0 + 0.7), 41),
                 np.linspace(h0 - 1.0, h0 + 1.0, 41)),
            ]
            truth = {"D_E": D0, "w_EE": w0, "h": h0}

            for a, b, avals, bvals in configs:
                Z = np.empty((len(bvals), len(avals)), dtype=np.float64)
                for j, bv in enumerate(bvals):
                    for i, av in enumerate(avals):
                        params = truth.copy(); params[a] = float(av); params[b] = float(bv)
                        Z[j, i] = residual_mse(E, dEdt, lap, params["D_E"], params["w_EE"], params["h"], tau)

                np.savez_compressed(figure_dir / f"landscape_{a}_{b}.npz", a=avals, b=bvals, loss=Z,
                                    a_name=a, b_name=b, a_true=truth[a], b_true=truth[b])
                fig, ax = plt.subplots(figsize=(6.2, 5.0))
                im = ax.imshow(np.log10(Z + 1e-16), origin="lower", aspect="auto",
                               extent=[avals[0], avals[-1], bvals[0], bvals[-1]])
                ax.scatter([truth[a]], [truth[b]], marker="*", s=150, label="Ground truth")
                ax.set_xlabel(a); ax.set_ylabel(b)
                ax.set_title(f"PDE residual landscape: {a} vs {b}")
                cbar = fig.colorbar(im, ax=ax); cbar.set_label("log10 residual MSE")
                ax.legend()
                fig.tight_layout()
                fig.savefig(figure_dir / f"landscape_{a}_{b}.png", dpi=300)
                fig.savefig(figure_dir / f"landscape_{a}_{b}.pdf")
                plt.close(fig)


        def xyt_for_frame(frame: int, T: int, H: int, W: int):
            x = np.linspace(-1, 1, W, dtype=np.float32)
            y = np.linspace(-1, 1, H, dtype=np.float32)
            yy, xx = np.meshgrid(y, x, indexing="ij")
            t = np.full_like(xx, -1.0 + 2.0 * frame / float(T - 1))
            return np.stack([xx.ravel(), yy.ravel(), t.ravel()], axis=1)


        def reconstruct_best_clean_run(rows, data_path: Path, output_dir: Path, figure_dir: Path):
            clean = [r for r in rows if abs(float(r["noise_sigma"])) < 1e-12]
            if not clean:
                return
            best = min(clean, key=lambda r: sum(r["relative_error_percent"][p] for p in ("D_E", "w_EE", "h")))
            ckpt_path = Path(best["checkpoint"])
            if not ckpt_path.is_absolute():
                # The training JSON generally contains a path relative to project working dir.
                candidate = output_dir.parent / ckpt_path
                if candidate.exists(): ckpt_path = candidate
                else: ckpt_path = output_dir / "checkpoints" / ckpt_path.name

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model = gPINN_WilsonCowan(fourier_seed=best.get("fourier_seed", 2718)).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            d = np.load(data_path)
            E = d["E"].astype(np.float32)
            T, H, W = E.shape
            for frame in (0, T // 2, T - 1):
                X = xyt_for_frame(frame, T, H, W)
                preds = []
                with torch.no_grad():
                    for start in range(0, len(X), 32768):
                        xb = torch.from_numpy(X[start:start+32768]).to(device)
                        preds.append(model.forward_E(xb).cpu().numpy())
                pred = np.concatenate(preds).reshape(H, W)
                err = np.abs(pred - E[frame])
                time_s = frame / 30.0
                for name, img, title in [
                    ("true", E[frame], f"Synthetic truth at t={time_s:.1f} s"),
                    ("pred", pred, f"gPINN reconstruction at t={time_s:.1f} s"),
                    ("abs_error", err, f"Absolute error at t={time_s:.1f} s"),
                ]:
                    fig, ax = plt.subplots(figsize=(5.3, 4.6))
                    im = ax.imshow(img, origin="lower")
                    ax.set_title(title); ax.set_xlabel("x pixel"); ax.set_ylabel("y pixel")
                    fig.colorbar(im, ax=ax)
                    fig.tight_layout()
                    fig.savefig(figure_dir / f"frame-{frame:03d}_{name}.png", dpi=300)
                    fig.savefig(figure_dir / f"frame-{frame:03d}_{name}.pdf")
                    plt.close(fig)


        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--output_dir", default="synthetic_outputs")
            p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
            args = p.parse_args()
            out = Path(args.output_dir)
            figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
            rows = load_jsons(out / "results")
            write_summary_csv(rows, out / "synthetic_recovery_runs.csv")
            aggregate_table(rows, out / "synthetic_recovery_summary.csv")
            for pnm in ("D_E", "w_EE", "h"):
                plot_parameter(rows, pnm, figdir)
            compute_and_plot_landscape(Path(args.data), figdir)
            reconstruct_best_clean_run(rows, Path(args.data), out, figdir)
            print(f"Wrote summaries and figures under: {out}")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# oracle: source train_synthetic_oracle_derivative.py
# ============================================================================
def oracle(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['train_synthetic_oracle_derivative.py'] + list(argv or [])
    try:
        """
        Synthetic oracle-derivative control for Wilson-Cowan gPINN parameter recovery.

        PURPOSE
        -------
        The current synthetic experiment showed:
          * excellent state reconstruction E,
          * poor E_t and Laplacian(E),
          * poor physical-parameter recovery,
          * yet near-exact direct recovery from the true synthetic trajectory.

        This control asks a causal question:
        "If the same neural representation is explicitly taught the correct derivatives
        on synthetic data, does physical-parameter recovery move toward ground truth?"

        This is intentionally an ORACLE SYNTHETIC CONTROL. The derivative targets come
        from the independently generated synthetic field and are NOT available for real
        calcium imaging. Do not present this as a real-data training recipe.

        Stages
        ------
        A. State + initial-condition pretraining.
        B. Sobolev/oracle derivative training using true E_t and Laplacian(E).
        C. Freeze the state network and optimize only D_E, w_EE, h from the learned
           field/derivatives.
        D. Joint low-learning-rate fine-tuning with state, derivative, and PDE losses.

        Outputs
        -------
        synthetic_outputs/oracle_derivative_control/
          seed-42_oracle_control.json
          seed-42_oracle_control.pt
          training_history.csv
          scatter_E.png
          scatter_Et.png
          scatter_laplacian.png
        """
        import argparse
        import csv
        import json
        import math
        import random
        from pathlib import Path

        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        from .model import gPINN_WilsonCowan


        # ------------------------------- utilities -------------------------------- #

        def set_seed(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


        def r2_score(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            ss_res = np.sum((a - b) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2)
            return float(1.0 - ss_res / (ss_tot + 1e-12))


        def corrcoef(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            if np.std(a) == 0 or np.std(b) == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])


        def metric_block(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            e = b - a
            return {
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "mae": float(np.mean(np.abs(e))),
                "r2": r2_score(a, b),
                "r": corrcoef(a, b),
                "relative_l2": float(np.linalg.norm(e) / (np.linalg.norm(a) + 1e-12)),
            }


        def sigmoid_np(z):
            z = np.clip(z, -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-z))


        def physical_parameter_dict(model):
            return {
                "D_E": float(model.forward_D().detach().cpu().item()),
                "w_EE": float(torch.clamp(model.w_EE, 0.5, 6.0).detach().cpu().item()),
                "h": float(model.h.detach().cpu().item()),
            }


        def relative_errors(truth, estimate):
            return {
                k: 100.0 * abs(estimate[k] - truth[k]) / (abs(truth[k]) + 1e-12)
                for k in truth
            }


        def find_raw_D_parameter(model):
            # Current package uses raw_D. Keep fallbacks for small naming changes.
            for name in ("raw_D", "D_raw", "raw_DE", "raw_d"):
                if hasattr(model, name):
                    return getattr(model, name), name
            # Last resort: inspect names.
            for name, p in model.named_parameters():
                if "raw" in name.lower() and "d" in name.lower():
                    return p, name
            raise AttributeError("Could not locate the raw D_E trainable parameter.")


        def normalized_mse(pred, target, eps=1e-8):
            mse = torch.mean((pred - target) ** 2)
            denom = torch.mean(target ** 2).detach() + eps
            return mse / denom


        # --------------------------- synthetic sampling ---------------------------- #

        def make_state_pool(E, n_points, seed, early_fraction=0.50, early_seconds=5.0,
                            fps=30.0):
            """
            Fixed state-training pool with deliberate oversampling of the early transient.
            """
            rng = np.random.default_rng(seed)
            T, H, W = E.shape
            n_early = int(round(n_points * early_fraction))
            n_uniform = n_points - n_early

            early_T = max(1, min(T, int(round(early_seconds * fps)) + 1))

            ti_e = rng.integers(0, early_T, size=n_early)
            yi_e = rng.integers(0, H, size=n_early)
            xi_e = rng.integers(0, W, size=n_early)

            ti_u = rng.integers(0, T, size=n_uniform)
            yi_u = rng.integers(0, H, size=n_uniform)
            xi_u = rng.integers(0, W, size=n_uniform)

            ti = np.concatenate([ti_e, ti_u])
            yi = np.concatenate([yi_e, yi_u])
            xi = np.concatenate([xi_e, xi_u])

            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -1.0 + 2.0 * ti / float(T - 1)

            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            target = E[ti, yi, xi].astype(np.float32)[:, None]
            return xyt, target


        def make_ic_pool(E, n_points, seed):
            rng = np.random.default_rng(seed)
            _, H, W = E.shape
            total = H * W
            n = min(n_points, total)
            flat = rng.choice(total, n, replace=False)
            yi = flat // W
            xi = flat % W
            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -np.ones_like(x)
            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            target = E[0, yi, xi].astype(np.float32)[:, None]
            return xyt, target


        def make_derivative_pool(E, l_fov, t_seconds, n_points, seed,
                                 early_fraction=0.60, early_seconds=5.0):
            """
            Interior samples with centered finite-difference E_t and Laplacian targets.
            More weight is deliberately placed on the early transient.
            """
            rng = np.random.default_rng(seed)
            T, H, W = E.shape
            dt = float(np.median(np.diff(t_seconds)))
            dx = float(l_fov) / float(W - 1)
            dy = float(l_fov) / float(H - 1)

            early_last = min(T - 2, max(1, int(round(early_seconds / dt))))
            n_early = int(round(n_points * early_fraction))
            n_uniform = n_points - n_early

            ti_e = rng.integers(1, early_last + 1, size=n_early)
            yi_e = rng.integers(1, H - 1, size=n_early)
            xi_e = rng.integers(1, W - 1, size=n_early)

            ti_u = rng.integers(1, T - 1, size=n_uniform)
            yi_u = rng.integers(1, H - 1, size=n_uniform)
            xi_u = rng.integers(1, W - 1, size=n_uniform)

            ti = np.concatenate([ti_e, ti_u])
            yi = np.concatenate([yi_e, yi_u])
            xi = np.concatenate([xi_e, xi_u])

            E0 = E[ti, yi, xi].astype(np.float64)
            dE_dt = (
                E[ti + 1, yi, xi].astype(np.float64)
                - E[ti - 1, yi, xi].astype(np.float64)
            ) / (2.0 * dt)

            d2x = (
                E[ti, yi, xi + 1].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi, xi - 1].astype(np.float64)
            ) / (dx * dx)
            d2y = (
                E[ti, yi + 1, xi].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi - 1, xi].astype(np.float64)
            ) / (dy * dy)
            lap = d2x + d2y

            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -1.0 + 2.0 * ti / float(T - 1)

            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            return (
                xyt,
                E0.astype(np.float32)[:, None],
                dE_dt.astype(np.float32)[:, None],
                lap.astype(np.float32)[:, None],
            )


        def batch_from_pool(pool_tensors, batch_size, generator=None):
            n = pool_tensors[0].shape[0]
            idx = torch.randint(
                0, n, (batch_size,),
                device=pool_tensors[0].device,
                generator=generator,
            )
            return tuple(x[idx] for x in pool_tensors)


        # ----------------------- autograd derivative functions --------------------- #

        def state_and_derivatives(model, xyt, duration, l_fov, create_graph=True):
            """
            Returns E, physical E_t, and physical Laplacian(E).
            xyt must be normalized to [-1,1]^3.
            """
            xyt = xyt.clone().detach().requires_grad_(True)
            E = model.forward_E(xyt)

            grad_E = torch.autograd.grad(
                E, xyt,
                grad_outputs=torch.ones_like(E),
                create_graph=True,
                retain_graph=True,
            )[0]

            dE_dx = grad_E[:, 0:1]
            dE_dy = grad_E[:, 1:2]
            dE_dt_norm = grad_E[:, 2:3]

            d2x = torch.autograd.grad(
                dE_dx, xyt,
                grad_outputs=torch.ones_like(dE_dx),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 0:1]

            d2y = torch.autograd.grad(
                dE_dy, xyt,
                grad_outputs=torch.ones_like(dE_dy),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 1:2]

            dE_dt = (2.0 / float(duration)) * dE_dt_norm
            lap = (4.0 / float(l_fov) ** 2) * (d2x + d2y)
            return E, dE_dt, lap


        def physics_residual(model, E, dE_dt, lap, tau):
            D = model.forward_D()
            w = torch.clamp(model.w_EE, 0.5, 6.0)
            h = model.h
            rhs = -E + torch.sigmoid(w * E + D * lap + h)
            return tau * dE_dt - rhs


        # ---------------------------- evaluation ---------------------------------- #

        @torch.no_grad()
        def predict_state(model, xyt, batch_size=4096):
            vals = []
            for s in range(0, len(xyt), batch_size):
                vals.append(
                    model.forward_E(xyt[s:s+batch_size]).detach().cpu().numpy().ravel()
                )
            return np.concatenate(vals)


        def predict_derivatives(model, xyt, duration, l_fov, batch_size=512):
            Es, Ts, Ls = [], [], []
            model.eval()
            for s in range(0, len(xyt), batch_size):
                E, Et, Lap = state_and_derivatives(
                    model, xyt[s:s+batch_size], duration, l_fov, create_graph=False
                )
                Es.append(E.detach().cpu().numpy().ravel())
                Ts.append(Et.detach().cpu().numpy().ravel())
                Ls.append(Lap.detach().cpu().numpy().ravel())
            return np.concatenate(Es), np.concatenate(Ts), np.concatenate(Ls)


        def scatter(x, y, xlabel, ylabel, title, path, max_points=12000):
            rng = np.random.default_rng(123)
            if len(x) > max_points:
                idx = rng.choice(len(x), max_points, replace=False)
                x, y = x[idx], y[idx]
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(x, y, s=5, alpha=0.25)
            lo = min(float(np.min(x)), float(np.min(y)))
            hi = max(float(np.max(x)), float(np.max(y)))
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)


        # ------------------------------- training --------------------------------- #

        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
            p.add_argument("--output_dir",
                           default="synthetic_outputs/oracle_derivative_control")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--fourier_seed", type=int, default=2718)

            p.add_argument("--state_pool", type=int, default=60000)
            p.add_argument("--deriv_pool", type=int, default=24000)
            p.add_argument("--ic_pool", type=int, default=8192)

            p.add_argument("--state_batch", type=int, default=4096)
            p.add_argument("--deriv_batch", type=int, default=512)
            p.add_argument("--ic_batch", type=int, default=2048)

            p.add_argument("--stageA_steps", type=int, default=2500)
            p.add_argument("--stageB_steps", type=int, default=2500)
            p.add_argument("--stageC_adam_steps", type=int, default=1800)
            p.add_argument("--stageD_steps", type=int, default=1200)

            p.add_argument("--lrA", type=float, default=1e-3)
            p.add_argument("--lrB", type=float, default=2e-4)
            p.add_argument("--lrC", type=float, default=1e-2)
            p.add_argument("--lrD_net", type=float, default=2e-5)
            p.add_argument("--lrD_phys", type=float, default=5e-4)

            p.add_argument("--duration", type=float, default=30.0)
            args = p.parse_args()

            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            set_seed(args.seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            d = np.load(args.data, allow_pickle=False)
            E_full = d["E"].astype(np.float32)
            t_seconds = d["t_seconds"].astype(np.float64)
            truth = {
                "D_E": float(d["D_E_true"]),
                "w_EE": float(d["w_EE_true"]),
                "h": float(d["h_true"]),
            }
            tau = float(d["tau"])
            l_fov = float(d["l_fov"])

            # Fixed pools.
            sx, sy = make_state_pool(
                E_full, args.state_pool, args.seed + 1,
                early_fraction=0.50, early_seconds=5.0
            )
            ix, iy = make_ic_pool(E_full, args.ic_pool, args.seed + 2)
            dx, dE, dEt, dLap = make_derivative_pool(
                E_full, l_fov, t_seconds, args.deriv_pool, args.seed + 3,
                early_fraction=0.60, early_seconds=5.0
            )

            S = (
                torch.from_numpy(sx).to(device),
                torch.from_numpy(sy).to(device),
            )
            I = (
                torch.from_numpy(ix).to(device),
                torch.from_numpy(iy).to(device),
            )
            DPOOL = (
                torch.from_numpy(dx).to(device),
                torch.from_numpy(dE).to(device),
                torch.from_numpy(dEt).to(device),
                torch.from_numpy(dLap).to(device),
            )

            model = gPINN_WilsonCowan(fourier_seed=args.fourier_seed).to(device)

            raw_D, raw_D_name = find_raw_D_parameter(model)
            phys_ids = {id(raw_D), id(model.w_EE), id(model.h)}
            network_params = [p for p in model.parameters() if id(p) not in phys_ids]
            physical_params = [raw_D, model.w_EE, model.h]

            history = []

            # ---------------- Stage A: state + IC pretraining ---------------- #
            print("\n=== Stage A: state + initial-condition pretraining ===", flush=True)
            for p_ in physical_params:
                p_.requires_grad_(False)
            for p_ in network_params:
                p_.requires_grad_(True)

            optA = torch.optim.Adam(network_params, lr=args.lrA)

            for step in range(1, args.stageA_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss = loss_data + 1.0 * loss_ic

                optA.zero_grad(set_to_none=True)
                loss.backward()
                optA.step()

                if step == 1 or step % 100 == 0 or step == args.stageA_steps:
                    print(
                        f"A {step:5d}/{args.stageA_steps} "
                        f"loss={loss.item():.4e} data={loss_data.item():.4e} "
                        f"ic={loss_ic.item():.4e}",
                        flush=True,
                    )
                    history.append(["A", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    "", "", ""])

            # -------------- Stage B: oracle derivative/Sobolev fit ----------- #
            print("\n=== Stage B: oracle derivative training ===", flush=True)
            optB = torch.optim.Adam(network_params, lr=args.lrB)

            for step in range(1, args.stageB_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)
                xd, Ed_true, Et_true, Lap_true = batch_from_pool(
                    DPOOL, args.deriv_batch
                )

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)
                Ed_pred, Et_pred, Lap_pred = state_and_derivatives(
                    model, xd, args.duration, l_fov, create_graph=True
                )

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss_Ed = normalized_mse(Ed_pred, Ed_true)
                loss_Et = normalized_mse(Et_pred, Et_true)
                loss_Lap = normalized_mse(Lap_pred, Lap_true)

                loss = (
                    1.0 * loss_data
                    + 0.75 * loss_ic
                    + 0.50 * loss_Ed
                    + 1.00 * loss_Et
                    + 1.00 * loss_Lap
                )

                optB.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network_params, max_norm=10.0)
                optB.step()

                if step == 1 or step % 50 == 0 or step == args.stageB_steps:
                    print(
                        f"B {step:5d}/{args.stageB_steps} loss={loss.item():.4e} "
                        f"data={loss_data.item():.3e} ic={loss_ic.item():.3e} "
                        f"E={loss_Ed.item():.3e} Et={loss_Et.item():.3e} "
                        f"Lap={loss_Lap.item():.3e}",
                        flush=True,
                    )
                    history.append(["B", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    float(loss_Et.item()), float(loss_Lap.item()), ""])

            # Evaluate derivative quality before parameter fitting.
            eval_n = min(12000, DPOOL[0].shape[0])
            eval_idx = torch.arange(eval_n, device=device)
            eval_x = DPOOL[0][eval_idx]
            eval_E_true = DPOOL[1][eval_idx].detach().cpu().numpy().ravel()
            eval_Et_true = DPOOL[2][eval_idx].detach().cpu().numpy().ravel()
            eval_Lap_true = DPOOL[3][eval_idx].detach().cpu().numpy().ravel()

            E_pre, Et_pre, Lap_pre = predict_derivatives(
                model, eval_x, args.duration, l_fov, batch_size=args.deriv_batch
            )
            pre_param_derivative_metrics = {
                "E": metric_block(eval_E_true, E_pre),
                "E_t": metric_block(eval_Et_true, Et_pre),
                "laplacian": metric_block(eval_Lap_true, Lap_pre),
            }
            print("\nDerivative metrics after Stage B:", flush=True)
            print(json.dumps(pre_param_derivative_metrics, indent=2), flush=True)

            # ---------------- Stage C: parameter-only optimization ------------ #
            print("\n=== Stage C: freeze network, optimize D_E / w_EE / h ===", flush=True)
            for p_ in network_params:
                p_.requires_grad_(False)
            for p_ in physical_params:
                p_.requires_grad_(True)

            # Cache learned E/derivatives on the derivative pool. No weight gradients now.
            cache_E, cache_Et, cache_Lap = [], [], []
            model.eval()
            for s in range(0, DPOOL[0].shape[0], args.deriv_batch):
                Ee, Ete, Lape = state_and_derivatives(
                    model, DPOOL[0][s:s+args.deriv_batch],
                    args.duration, l_fov, create_graph=False
                )
                cache_E.append(Ee.detach())
                cache_Et.append(Ete.detach())
                cache_Lap.append(Lape.detach())

            cache_E = torch.cat(cache_E, dim=0)
            cache_Et = torch.cat(cache_Et, dim=0)
            cache_Lap = torch.cat(cache_Lap, dim=0)

            optC = torch.optim.Adam(physical_params, lr=args.lrC)

            for step in range(1, args.stageC_adam_steps + 1):
                n = cache_E.shape[0]
                idx = torch.randint(0, n, (min(8192, n),), device=device)
                residual = physics_residual(
                    model,
                    cache_E[idx],
                    cache_Et[idx],
                    cache_Lap[idx],
                    tau,
                )
                loss_phys = torch.mean(residual ** 2)

                optC.zero_grad(set_to_none=True)
                loss_phys.backward()
                optC.step()

                # Mirror the hard w_EE constraint used by the inverse pipeline.
                with torch.no_grad():
                    model.w_EE.clamp_(0.5, 6.0)

                if step == 1 or step % 100 == 0 or step == args.stageC_adam_steps:
                    est = physical_parameter_dict(model)
                    print(
                        f"C {step:5d}/{args.stageC_adam_steps} "
                        f"physics={loss_phys.item():.4e} "
                        f"D={est['D_E']:.6f} w={est['w_EE']:.6f} h={est['h']:.6f}",
                        flush=True,
                    )
                    history.append(["C", step, float(loss_phys.item()),
                                    "", "", "", "", json.dumps(est)])

            stageC_estimate = physical_parameter_dict(model)
            print("Stage C estimate:", json.dumps(stageC_estimate, indent=2), flush=True)

            # ---------------- Stage D: joint low-LR fine tuning --------------- #
            print("\n=== Stage D: joint derivative-aware physics fine-tuning ===", flush=True)
            for p_ in network_params:
                p_.requires_grad_(True)
            for p_ in physical_params:
                p_.requires_grad_(True)

            optD = torch.optim.Adam([
                {"params": network_params, "lr": args.lrD_net},
                {"params": physical_params, "lr": args.lrD_phys},
            ])

            for step in range(1, args.stageD_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)
                xd, Ed_true, Et_true, Lap_true = batch_from_pool(
                    DPOOL, args.deriv_batch
                )

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)
                Ed_pred, Et_pred, Lap_pred = state_and_derivatives(
                    model, xd, args.duration, l_fov, create_graph=True
                )
                residual = physics_residual(
                    model, Ed_pred, Et_pred, Lap_pred, tau
                )

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss_Ed = normalized_mse(Ed_pred, Ed_true)
                loss_Et = normalized_mse(Et_pred, Et_true)
                loss_Lap = normalized_mse(Lap_pred, Lap_true)

                # Scale the PDE residual by a fixed characteristic scale so it contributes
                # without overwhelming the Sobolev supervision.
                loss_phys = torch.mean(residual ** 2) / 1e-3

                loss = (
                    1.0 * loss_data
                    + 0.50 * loss_ic
                    + 0.50 * loss_Ed
                    + 1.00 * loss_Et
                    + 1.00 * loss_Lap
                    + 0.25 * loss_phys
                )

                optD.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optD.step()

                with torch.no_grad():
                    model.w_EE.clamp_(0.5, 6.0)

                if step == 1 or step % 50 == 0 or step == args.stageD_steps:
                    est = physical_parameter_dict(model)
                    print(
                        f"D {step:5d}/{args.stageD_steps} loss={loss.item():.4e} "
                        f"Et={loss_Et.item():.3e} Lap={loss_Lap.item():.3e} "
                        f"phys={loss_phys.item():.3e} "
                        f"D={est['D_E']:.6f} w={est['w_EE']:.6f} h={est['h']:.6f}",
                        flush=True,
                    )
                    history.append(["D", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    float(loss_Et.item()), float(loss_Lap.item()),
                                    json.dumps(est)])

            # --------------------------- final evaluation ---------------------- #
            model.eval()
            final_E, final_Et, final_Lap = predict_derivatives(
                model, eval_x, args.duration, l_fov, batch_size=args.deriv_batch
            )

            final_derivative_metrics = {
                "E": metric_block(eval_E_true, final_E),
                "E_t": metric_block(eval_Et_true, final_Et),
                "laplacian": metric_block(eval_Lap_true, final_Lap),
            }

            final_estimate = physical_parameter_dict(model)
            final_relerr = relative_errors(truth, final_estimate)

            # Residual on learned field.
            res_learned = (
                tau * final_Et
                - (
                    -final_E
                    + sigmoid_np(
                        final_estimate["w_EE"] * final_E
                        + final_estimate["D_E"] * final_Lap
                        + final_estimate["h"]
                    )
                )
            )

            report = {
                "experiment": "synthetic_oracle_derivative_control",
                "important_note": (
                    "Derivative supervision uses finite-difference derivatives of the "
                    "known synthetic truth and is an oracle diagnostic, not a real-data recipe."
                ),
                "seed": args.seed,
                "fourier_seed": args.fourier_seed,
                "device": str(device),
                "truth": truth,
                "initial_parameterization": {
                    "D_E": float(0.1 + 4.9 * 0.5),
                    "w_EE": 2.0,
                    "h": -1.0,
                },
                "raw_D_parameter_name": raw_D_name,
                "after_stageB_derivative_metrics": pre_param_derivative_metrics,
                "stageC_parameter_only_estimate": stageC_estimate,
                "final_estimate": final_estimate,
                "final_relative_error_percent": final_relerr,
                "final_derivative_metrics": final_derivative_metrics,
                "final_learned_field_pde_residual_mse": float(np.mean(res_learned ** 2)),
                "settings": vars(args),
            }

            print("\n=== FINAL REPORT ===", flush=True)
            print(json.dumps(report, indent=2), flush=True)

            tag = f"seed-{args.seed}_oracle_control"
            (out / f"{tag}.json").write_text(json.dumps(report, indent=2))

            torch.save({
                "model_state": model.state_dict(),
                "report": report,
            }, out / f"{tag}.pt")

            with open(out / "training_history.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "stage", "step", "loss", "data_loss", "ic_loss",
                    "Et_loss", "lap_loss", "parameter_estimate"
                ])
                writer.writerows(history)

            scatter(
                eval_E_true, final_E,
                "Synthetic truth E", "Oracle-control gPINN E",
                "State accuracy after oracle derivative control",
                out / "scatter_E.png",
            )
            scatter(
                eval_Et_true, final_Et,
                "Finite-difference truth E_t", "Oracle-control gPINN E_t",
                "Time-derivative accuracy after oracle control",
                out / "scatter_Et.png",
            )
            scatter(
                eval_Lap_true, final_Lap,
                "Finite-difference truth Laplacian(E)",
                "Oracle-control gPINN Laplacian(E)",
                "Laplacian accuracy after oracle control",
                out / "scatter_laplacian.png",
            )

            print(f"\nSaved outputs to {out}", flush=True)
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# staged: source train_synthetic_staged_nonoracle.py
# ============================================================================
def staged(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['train_synthetic_staged_nonoracle.py'] + list(argv or [])
    try:
        """
        Synthetic staged non-oracle control for Wilson-Cowan gPINN parameter recovery.

        PURPOSE
        -------
        The current synthetic experiment showed:
          * excellent state reconstruction E,
          * poor E_t and Laplacian(E),
          * poor physical-parameter recovery,
          * yet near-exact direct recovery from the true synthetic trajectory.

        This control asks a causal question:
        "If the same neural representation is explicitly taught the correct derivatives
        on synthetic data, does physical-parameter recovery move toward ground truth?"

        This is intentionally an ORACLE SYNTHETIC CONTROL. The derivative targets come
        from the independently generated synthetic field and are NOT available for real
        calcium imaging. Do not present this as a real-data training recipe.

        Stages
        ------
        A. State + initial-condition pretraining.
        B. Sobolev/oracle derivative training using true E_t and Laplacian(E).
        C. Freeze the state network and optimize only D_E, w_EE, h from the learned
           field/derivatives.
        D. Joint low-learning-rate fine-tuning with state, derivative, and PDE losses.

        Outputs
        -------
        synthetic_outputs/staged_nonoracle_control/
          seed-42_staged_nonoracle.json
          seed-42_staged_nonoracle.pt
          training_history.csv
          scatter_E.png
          scatter_Et.png
          scatter_laplacian.png
        """
        import argparse
        import csv
        import json
        import math
        import random
        from pathlib import Path

        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        from .model import gPINN_WilsonCowan


        # ------------------------------- utilities -------------------------------- #

        def set_seed(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


        def r2_score(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            ss_res = np.sum((a - b) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2)
            return float(1.0 - ss_res / (ss_tot + 1e-12))


        def corrcoef(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            if np.std(a) == 0 or np.std(b) == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])


        def metric_block(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            e = b - a
            return {
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "mae": float(np.mean(np.abs(e))),
                "r2": r2_score(a, b),
                "r": corrcoef(a, b),
                "relative_l2": float(np.linalg.norm(e) / (np.linalg.norm(a) + 1e-12)),
            }


        def sigmoid_np(z):
            z = np.clip(z, -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-z))


        def physical_parameter_dict(model):
            return {
                "D_E": float(model.forward_D().detach().cpu().item()),
                "w_EE": float(torch.clamp(model.w_EE, 0.5, 6.0).detach().cpu().item()),
                "h": float(model.h.detach().cpu().item()),
            }


        def relative_errors(truth, estimate):
            return {
                k: 100.0 * abs(estimate[k] - truth[k]) / (abs(truth[k]) + 1e-12)
                for k in truth
            }


        def find_raw_D_parameter(model):
            # Current package uses raw_D. Keep fallbacks for small naming changes.
            for name in ("raw_D", "D_raw", "raw_DE", "raw_d"):
                if hasattr(model, name):
                    return getattr(model, name), name
            # Last resort: inspect names.
            for name, p in model.named_parameters():
                if "raw" in name.lower() and "d" in name.lower():
                    return p, name
            raise AttributeError("Could not locate the raw D_E trainable parameter.")


        def normalized_mse(pred, target, eps=1e-8):
            mse = torch.mean((pred - target) ** 2)
            denom = torch.mean(target ** 2).detach() + eps
            return mse / denom


        # --------------------------- synthetic sampling ---------------------------- #

        def make_state_pool(E, n_points, seed, early_fraction=0.50, early_seconds=5.0,
                            fps=30.0):
            """
            Fixed state-training pool with deliberate oversampling of the early transient.
            """
            rng = np.random.default_rng(seed)
            T, H, W = E.shape
            n_early = int(round(n_points * early_fraction))
            n_uniform = n_points - n_early

            early_T = max(1, min(T, int(round(early_seconds * fps)) + 1))

            ti_e = rng.integers(0, early_T, size=n_early)
            yi_e = rng.integers(0, H, size=n_early)
            xi_e = rng.integers(0, W, size=n_early)

            ti_u = rng.integers(0, T, size=n_uniform)
            yi_u = rng.integers(0, H, size=n_uniform)
            xi_u = rng.integers(0, W, size=n_uniform)

            ti = np.concatenate([ti_e, ti_u])
            yi = np.concatenate([yi_e, yi_u])
            xi = np.concatenate([xi_e, xi_u])

            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -1.0 + 2.0 * ti / float(T - 1)

            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            target = E[ti, yi, xi].astype(np.float32)[:, None]
            return xyt, target


        def make_ic_pool(E, n_points, seed):
            rng = np.random.default_rng(seed)
            _, H, W = E.shape
            total = H * W
            n = min(n_points, total)
            flat = rng.choice(total, n, replace=False)
            yi = flat // W
            xi = flat % W
            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -np.ones_like(x)
            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            target = E[0, yi, xi].astype(np.float32)[:, None]
            return xyt, target


        def make_derivative_pool(E, l_fov, t_seconds, n_points, seed,
                                 early_fraction=0.60, early_seconds=5.0):
            """
            Interior samples with centered finite-difference E_t and Laplacian targets.
            More weight is deliberately placed on the early transient.
            """
            rng = np.random.default_rng(seed)
            T, H, W = E.shape
            dt = float(np.median(np.diff(t_seconds)))
            dx = float(l_fov) / float(W - 1)
            dy = float(l_fov) / float(H - 1)

            early_last = min(T - 2, max(1, int(round(early_seconds / dt))))
            n_early = int(round(n_points * early_fraction))
            n_uniform = n_points - n_early

            ti_e = rng.integers(1, early_last + 1, size=n_early)
            yi_e = rng.integers(1, H - 1, size=n_early)
            xi_e = rng.integers(1, W - 1, size=n_early)

            ti_u = rng.integers(1, T - 1, size=n_uniform)
            yi_u = rng.integers(1, H - 1, size=n_uniform)
            xi_u = rng.integers(1, W - 1, size=n_uniform)

            ti = np.concatenate([ti_e, ti_u])
            yi = np.concatenate([yi_e, yi_u])
            xi = np.concatenate([xi_e, xi_u])

            E0 = E[ti, yi, xi].astype(np.float64)
            dE_dt = (
                E[ti + 1, yi, xi].astype(np.float64)
                - E[ti - 1, yi, xi].astype(np.float64)
            ) / (2.0 * dt)

            d2x = (
                E[ti, yi, xi + 1].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi, xi - 1].astype(np.float64)
            ) / (dx * dx)
            d2y = (
                E[ti, yi + 1, xi].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi - 1, xi].astype(np.float64)
            ) / (dy * dy)
            lap = d2x + d2y

            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -1.0 + 2.0 * ti / float(T - 1)

            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            return (
                xyt,
                E0.astype(np.float32)[:, None],
                dE_dt.astype(np.float32)[:, None],
                lap.astype(np.float32)[:, None],
            )


        def batch_from_pool(pool_tensors, batch_size, generator=None):
            n = pool_tensors[0].shape[0]
            idx = torch.randint(
                0, n, (batch_size,),
                device=pool_tensors[0].device,
                generator=generator,
            )
            return tuple(x[idx] for x in pool_tensors)


        # ----------------------- autograd derivative functions --------------------- #

        def state_and_derivatives(model, xyt, duration, l_fov, create_graph=True):
            """
            Returns E, physical E_t, and physical Laplacian(E).
            xyt must be normalized to [-1,1]^3.
            """
            xyt = xyt.clone().detach().requires_grad_(True)
            E = model.forward_E(xyt)

            grad_E = torch.autograd.grad(
                E, xyt,
                grad_outputs=torch.ones_like(E),
                create_graph=True,
                retain_graph=True,
            )[0]

            dE_dx = grad_E[:, 0:1]
            dE_dy = grad_E[:, 1:2]
            dE_dt_norm = grad_E[:, 2:3]

            d2x = torch.autograd.grad(
                dE_dx, xyt,
                grad_outputs=torch.ones_like(dE_dx),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 0:1]

            d2y = torch.autograd.grad(
                dE_dy, xyt,
                grad_outputs=torch.ones_like(dE_dy),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 1:2]

            dE_dt = (2.0 / float(duration)) * dE_dt_norm
            lap = (4.0 / float(l_fov) ** 2) * (d2x + d2y)
            return E, dE_dt, lap


        def physics_residual(model, E, dE_dt, lap, tau):
            D = model.forward_D()
            w = torch.clamp(model.w_EE, 0.5, 6.0)
            h = model.h
            rhs = -E + torch.sigmoid(w * E + D * lap + h)
            return tau * dE_dt - rhs


        # ---------------------------- evaluation ---------------------------------- #

        @torch.no_grad()
        def predict_state(model, xyt, batch_size=4096):
            vals = []
            for s in range(0, len(xyt), batch_size):
                vals.append(
                    model.forward_E(xyt[s:s+batch_size]).detach().cpu().numpy().ravel()
                )
            return np.concatenate(vals)


        def predict_derivatives(model, xyt, duration, l_fov, batch_size=512):
            Es, Ts, Ls = [], [], []
            model.eval()
            for s in range(0, len(xyt), batch_size):
                E, Et, Lap = state_and_derivatives(
                    model, xyt[s:s+batch_size], duration, l_fov, create_graph=False
                )
                Es.append(E.detach().cpu().numpy().ravel())
                Ts.append(Et.detach().cpu().numpy().ravel())
                Ls.append(Lap.detach().cpu().numpy().ravel())
            return np.concatenate(Es), np.concatenate(Ts), np.concatenate(Ls)


        def scatter(x, y, xlabel, ylabel, title, path, max_points=12000):
            rng = np.random.default_rng(123)
            if len(x) > max_points:
                idx = rng.choice(len(x), max_points, replace=False)
                x, y = x[idx], y[idx]
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(x, y, s=5, alpha=0.25)
            lo = min(float(np.min(x)), float(np.min(y)))
            hi = max(float(np.max(x)), float(np.max(y)))
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)


        # ------------------------------- training --------------------------------- #

        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
            p.add_argument("--output_dir",
                           default="synthetic_outputs/staged_nonoracle_control")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--fourier_seed", type=int, default=2718)

            p.add_argument("--state_pool", type=int, default=60000)
            p.add_argument("--deriv_pool", type=int, default=24000)
            p.add_argument("--ic_pool", type=int, default=8192)

            p.add_argument("--state_batch", type=int, default=4096)
            p.add_argument("--deriv_batch", type=int, default=512)
            p.add_argument("--ic_batch", type=int, default=2048)

            p.add_argument("--stageA_steps", type=int, default=2500)
            p.add_argument("--stageB_steps", type=int, default=2500)
            p.add_argument("--stageC_adam_steps", type=int, default=1800)
            p.add_argument("--stageD_steps", type=int, default=1200)

            p.add_argument("--lrA", type=float, default=1e-3)
            p.add_argument("--lrB", type=float, default=2e-4)
            p.add_argument("--lrC", type=float, default=1e-2)
            p.add_argument("--lrD_net", type=float, default=2e-5)
            p.add_argument("--lrD_phys", type=float, default=5e-4)

            p.add_argument("--duration", type=float, default=30.0)
            args = p.parse_args()

            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            set_seed(args.seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            d = np.load(args.data, allow_pickle=False)
            E_full = d["E"].astype(np.float32)
            t_seconds = d["t_seconds"].astype(np.float64)
            truth = {
                "D_E": float(d["D_E_true"]),
                "w_EE": float(d["w_EE_true"]),
                "h": float(d["h_true"]),
            }
            tau = float(d["tau"])
            l_fov = float(d["l_fov"])

            # Fixed pools.
            sx, sy = make_state_pool(
                E_full, args.state_pool, args.seed + 1,
                early_fraction=0.50, early_seconds=5.0
            )
            ix, iy = make_ic_pool(E_full, args.ic_pool, args.seed + 2)
            dx, dE, dEt, dLap = make_derivative_pool(
                E_full, l_fov, t_seconds, args.deriv_pool, args.seed + 3,
                early_fraction=0.60, early_seconds=5.0
            )

            S = (
                torch.from_numpy(sx).to(device),
                torch.from_numpy(sy).to(device),
            )
            I = (
                torch.from_numpy(ix).to(device),
                torch.from_numpy(iy).to(device),
            )
            DPOOL = (
                torch.from_numpy(dx).to(device),
                torch.from_numpy(dE).to(device),
                torch.from_numpy(dEt).to(device),
                torch.from_numpy(dLap).to(device),
            )

            model = gPINN_WilsonCowan(fourier_seed=args.fourier_seed).to(device)

            raw_D, raw_D_name = find_raw_D_parameter(model)
            phys_ids = {id(raw_D), id(model.w_EE), id(model.h)}
            network_params = [p for p in model.parameters() if id(p) not in phys_ids]
            physical_params = [raw_D, model.w_EE, model.h]

            history = []

            # ---------------- Stage A: state + IC pretraining ---------------- #
            print("\n=== Stage A: state + initial-condition pretraining ===", flush=True)
            for p_ in physical_params:
                p_.requires_grad_(False)
            for p_ in network_params:
                p_.requires_grad_(True)

            optA = torch.optim.Adam(network_params, lr=args.lrA)

            for step in range(1, args.stageA_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss = loss_data + 1.0 * loss_ic

                optA.zero_grad(set_to_none=True)
                loss.backward()
                optA.step()

                if step == 1 or step % 100 == 0 or step == args.stageA_steps:
                    print(
                        f"A {step:5d}/{args.stageA_steps} "
                        f"loss={loss.item():.4e} data={loss_data.item():.4e} "
                        f"ic={loss_ic.item():.4e}",
                        flush=True,
                    )
                    history.append(["A", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    "", "", ""])

            # -------------- Stage B: matched non-oracle state/IC fit ----------- #
            print("\n=== Stage B: matched non-oracle state + IC training ===", flush=True)
            optB = torch.optim.Adam(network_params, lr=args.lrB)

            for step in range(1, args.stageB_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss = 1.0 * loss_data + 0.75 * loss_ic

                optB.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network_params, max_norm=10.0)
                optB.step()

                if step == 1 or step % 50 == 0 or step == args.stageB_steps:
                    print(
                        f"B {step:5d}/{args.stageB_steps} loss={loss.item():.4e} "
                        f"data={loss_data.item():.3e} ic={loss_ic.item():.3e}",
                        flush=True,
                    )
                    history.append(["B", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    "", "", ""])

            # Evaluate derivative quality before parameter fitting.
            eval_n = min(12000, DPOOL[0].shape[0])
            eval_idx = torch.arange(eval_n, device=device)
            eval_x = DPOOL[0][eval_idx]
            eval_E_true = DPOOL[1][eval_idx].detach().cpu().numpy().ravel()
            eval_Et_true = DPOOL[2][eval_idx].detach().cpu().numpy().ravel()
            eval_Lap_true = DPOOL[3][eval_idx].detach().cpu().numpy().ravel()

            E_pre, Et_pre, Lap_pre = predict_derivatives(
                model, eval_x, args.duration, l_fov, batch_size=args.deriv_batch
            )
            pre_param_derivative_metrics = {
                "E": metric_block(eval_E_true, E_pre),
                "E_t": metric_block(eval_Et_true, Et_pre),
                "laplacian": metric_block(eval_Lap_true, Lap_pre),
            }
            print("\nDerivative metrics after non-oracle Stage B:", flush=True)
            print(json.dumps(pre_param_derivative_metrics, indent=2), flush=True)

            # ---------------- Stage C: parameter-only optimization ------------ #
            print("\n=== Stage C: freeze network, optimize D_E / w_EE / h ===", flush=True)
            for p_ in network_params:
                p_.requires_grad_(False)
            for p_ in physical_params:
                p_.requires_grad_(True)

            # Cache learned E/derivatives on the derivative pool. No weight gradients now.
            cache_E, cache_Et, cache_Lap = [], [], []
            model.eval()
            for s in range(0, DPOOL[0].shape[0], args.deriv_batch):
                Ee, Ete, Lape = state_and_derivatives(
                    model, DPOOL[0][s:s+args.deriv_batch],
                    args.duration, l_fov, create_graph=False
                )
                cache_E.append(Ee.detach())
                cache_Et.append(Ete.detach())
                cache_Lap.append(Lape.detach())

            cache_E = torch.cat(cache_E, dim=0)
            cache_Et = torch.cat(cache_Et, dim=0)
            cache_Lap = torch.cat(cache_Lap, dim=0)

            optC = torch.optim.Adam(physical_params, lr=args.lrC)

            for step in range(1, args.stageC_adam_steps + 1):
                n = cache_E.shape[0]
                idx = torch.randint(0, n, (min(8192, n),), device=device)
                residual = physics_residual(
                    model,
                    cache_E[idx],
                    cache_Et[idx],
                    cache_Lap[idx],
                    tau,
                )
                loss_phys = torch.mean(residual ** 2)

                optC.zero_grad(set_to_none=True)
                loss_phys.backward()
                optC.step()

                # Mirror the hard w_EE constraint used by the inverse pipeline.
                with torch.no_grad():
                    model.w_EE.clamp_(0.5, 6.0)

                if step == 1 or step % 100 == 0 or step == args.stageC_adam_steps:
                    est = physical_parameter_dict(model)
                    print(
                        f"C {step:5d}/{args.stageC_adam_steps} "
                        f"physics={loss_phys.item():.4e} "
                        f"D={est['D_E']:.6f} w={est['w_EE']:.6f} h={est['h']:.6f}",
                        flush=True,
                    )
                    history.append(["C", step, float(loss_phys.item()),
                                    "", "", "", "", json.dumps(est)])

            stageC_estimate = physical_parameter_dict(model)
            print("Stage C estimate:", json.dumps(stageC_estimate, indent=2), flush=True)

            # ---------------- Stage D: joint low-LR non-oracle fine tuning ------ #
            print("\n=== Stage D: joint non-oracle physics fine-tuning ===", flush=True)
            for p_ in network_params:
                p_.requires_grad_(True)
            for p_ in physical_params:
                p_.requires_grad_(True)

            optD = torch.optim.Adam([
                {"params": network_params, "lr": args.lrD_net},
                {"params": physical_params, "lr": args.lrD_phys},
            ])

            for step in range(1, args.stageD_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)
                xd, _, _, _ = batch_from_pool(DPOOL, args.deriv_batch)

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)
                Ed_pred, Et_pred, Lap_pred = state_and_derivatives(
                    model, xd, args.duration, l_fov, create_graph=True
                )
                residual = physics_residual(model, Ed_pred, Et_pred, Lap_pred, tau)

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss_phys = torch.mean(residual ** 2) / 1e-3

                loss = 1.0 * loss_data + 0.50 * loss_ic + 0.25 * loss_phys

                optD.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optD.step()

                with torch.no_grad():
                    model.w_EE.clamp_(0.5, 6.0)

                if step == 1 or step % 50 == 0 or step == args.stageD_steps:
                    est = physical_parameter_dict(model)
                    print(
                        f"D {step:5d}/{args.stageD_steps} loss={loss.item():.4e} "
                        f"phys={loss_phys.item():.3e} "
                        f"D={est['D_E']:.6f} w={est['w_EE']:.6f} h={est['h']:.6f}",
                        flush=True,
                    )
                    history.append(["D", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    "", "", json.dumps(est)])

            # --------------------------- final evaluation ---------------------- #
            model.eval()
            final_E, final_Et, final_Lap = predict_derivatives(
                model, eval_x, args.duration, l_fov, batch_size=args.deriv_batch
            )

            final_derivative_metrics = {
                "E": metric_block(eval_E_true, final_E),
                "E_t": metric_block(eval_Et_true, final_Et),
                "laplacian": metric_block(eval_Lap_true, final_Lap),
            }

            final_estimate = physical_parameter_dict(model)
            final_relerr = relative_errors(truth, final_estimate)

            # Residual on learned field.
            res_learned = (
                tau * final_Et
                - (
                    -final_E
                    + sigmoid_np(
                        final_estimate["w_EE"] * final_E
                        + final_estimate["D_E"] * final_Lap
                        + final_estimate["h"]
                    )
                )
            )

            report = {
                "experiment": "synthetic_staged_nonoracle_control",
                "important_note": (
                    "Matched staged control with no oracle derivative targets. "
                    "Derivative metrics are evaluated only after training for comparison."
                ),
                "seed": args.seed,
                "fourier_seed": args.fourier_seed,
                "device": str(device),
                "truth": truth,
                "initial_parameterization": {
                    "D_E": float(0.1 + 4.9 * 0.5),
                    "w_EE": 2.0,
                    "h": -1.0,
                },
                "raw_D_parameter_name": raw_D_name,
                "after_stageB_diagnostic_derivative_metrics": pre_param_derivative_metrics,
                "stageC_parameter_only_estimate": stageC_estimate,
                "final_estimate": final_estimate,
                "final_relative_error_percent": final_relerr,
                "final_derivative_metrics": final_derivative_metrics,
                "final_learned_field_pde_residual_mse": float(np.mean(res_learned ** 2)),
                "settings": vars(args),
            }

            print("\n=== FINAL REPORT ===", flush=True)
            print(json.dumps(report, indent=2), flush=True)

            tag = f"seed-{args.seed}_staged_nonoracle"
            (out / f"{tag}.json").write_text(json.dumps(report, indent=2))

            torch.save({
                "model_state": model.state_dict(),
                "report": report,
            }, out / f"{tag}.pt")

            with open(out / "training_history.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "stage", "step", "loss", "data_loss", "ic_loss",
                    "Et_loss", "lap_loss", "parameter_estimate"
                ])
                writer.writerows(history)

            scatter(
                eval_E_true, final_E,
                "Synthetic truth E", "Staged non-oracle gPINN E",
                "State accuracy after staged non-oracle control",
                out / "scatter_E.png",
            )
            scatter(
                eval_Et_true, final_Et,
                "Finite-difference truth E_t", "Staged non-oracle gPINN E_t",
                "Time-derivative accuracy after staged non-oracle control",
                out / "scatter_Et.png",
            )
            scatter(
                eval_Lap_true, final_Lap,
                "Finite-difference truth Laplacian(E)",
                "Staged non-oracle gPINN Laplacian(E)",
                "Laplacian accuracy after staged non-oracle control",
                out / "scatter_laplacian.png",
            )

            print(f"\nSaved outputs to {out}", flush=True)
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# pseudoderiv: source train_synthetic_pseudoderiv_regression.py
# ============================================================================
def pseudoderiv(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['train_synthetic_pseudoderiv_regression.py'] + list(argv or [])
    try:
        """
        Synthetic real-compatible pseudo-derivative + regression control for Wilson-Cowan gPINN parameter recovery.

        PURPOSE
        -------
        The current synthetic experiment showed:
          * excellent state reconstruction E,
          * poor E_t and Laplacian(E),
          * poor physical-parameter recovery,
          * yet near-exact direct recovery from the true synthetic trajectory.

        This experiment replaces oracle derivatives with derivatives estimated from the
        observed field after mild spatiotemporal Gaussian smoothing. These pseudo-derivatives
        are available in principle for real calcium imaging. A bounded logit-regression estimate
        is also used to initialize the physical parameters before physics refinement.

        Stages
        ------
        A. State + initial-condition pretraining.
        B. Pseudo-derivative training using smoothed-observation E_t and Laplacian(E).
        C. Freeze the state network and optimize only D_E, w_EE, h from the learned
           field/derivatives.
        D. Joint low-learning-rate fine-tuning with state, derivative, and PDE losses.

        Outputs
        -------
        synthetic_outputs/pseudoderivative_regression_control/
          seed-42_pseudoderiv_regression.json
          seed-42_pseudoderiv_regression.pt
          training_history.csv
          scatter_E.png
          scatter_Et.png
          scatter_laplacian.png
        """
        import argparse
        import csv
        import json
        import math
        import random
        from pathlib import Path

        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        from scipy.optimize import lsq_linear

        from .model import gPINN_WilsonCowan


        # ------------------------------- utilities -------------------------------- #

        def set_seed(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


        def r2_score(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            ss_res = np.sum((a - b) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2)
            return float(1.0 - ss_res / (ss_tot + 1e-12))


        def corrcoef(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            if np.std(a) == 0 or np.std(b) == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])


        def metric_block(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            e = b - a
            return {
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "mae": float(np.mean(np.abs(e))),
                "r2": r2_score(a, b),
                "r": corrcoef(a, b),
                "relative_l2": float(np.linalg.norm(e) / (np.linalg.norm(a) + 1e-12)),
            }


        def sigmoid_np(z):
            z = np.clip(z, -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-z))


        def physical_parameter_dict(model):
            return {
                "D_E": float(model.forward_D().detach().cpu().item()),
                "w_EE": float(torch.clamp(model.w_EE, 0.5, 6.0).detach().cpu().item()),
                "h": float(model.h.detach().cpu().item()),
            }


        def relative_errors(truth, estimate):
            return {
                k: 100.0 * abs(estimate[k] - truth[k]) / (abs(truth[k]) + 1e-12)
                for k in truth
            }


        def find_raw_D_parameter(model):
            # Current package uses raw_D. Keep fallbacks for small naming changes.
            for name in ("raw_D", "D_raw", "raw_DE", "raw_d"):
                if hasattr(model, name):
                    return getattr(model, name), name
            # Last resort: inspect names.
            for name, p in model.named_parameters():
                if "raw" in name.lower() and "d" in name.lower():
                    return p, name
            raise AttributeError("Could not locate the raw D_E trainable parameter.")


        def normalized_mse(pred, target, eps=1e-8):
            mse = torch.mean((pred - target) ** 2)
            denom = torch.mean(target ** 2).detach() + eps
            return mse / denom



        def bounded_logit_regression(E, dE_dt, lap, tau, eps=1e-4):
            """
            Real-compatible initialization:
                logit(tau E_t + E) = w_EE E + D_E Lap(E) + h

            Uses bounded least squares for w_EE and D_E. The derivatives supplied here
            are pseudo-derivatives estimated from the smoothed observed field.
            """
            E = np.asarray(E).ravel().astype(np.float64)
            dE_dt = np.asarray(dE_dt).ravel().astype(np.float64)
            lap = np.asarray(lap).ravel().astype(np.float64)

            q = tau * dE_dt + E
            valid = np.isfinite(q) & np.isfinite(E) & np.isfinite(lap)
            valid &= (q > eps) & (q < 1.0 - eps)

            qv = q[valid]
            Ev = E[valid]
            Lv = lap[valid]
            y = np.log(qv / (1.0 - qv))
            X = np.column_stack([Ev, Lv, np.ones_like(Ev)])

            fit = lsq_linear(
                X, y,
                bounds=(
                    np.array([0.5, 0.1, -np.inf]),
                    np.array([6.0, 5.0,  np.inf]),
                ),
                method="trf",
                lsmr_tol="auto",
                verbose=0,
            )
            w_hat, D_hat, h_hat = map(float, fit.x)

            Xstd = np.column_stack([
                (Ev - Ev.mean()) / (Ev.std() + 1e-12),
                (Lv - Lv.mean()) / (Lv.std() + 1e-12),
            ])

            return {
                "n_valid": int(valid.sum()),
                "fraction_valid": float(valid.mean()),
                "estimate": {"D_E": D_hat, "w_EE": w_hat, "h": h_hat},
                "cost": float(fit.cost),
                "optimality": float(fit.optimality),
                "success": bool(fit.success),
                "condition_number_raw_X": float(np.linalg.cond(X)),
                "condition_number_standardized_E_lap": float(np.linalg.cond(Xstd)),
                "corr_E_laplacian": float(np.corrcoef(Ev, Lv)[0, 1]),
            }


        def set_physical_parameters(model, raw_D, estimate):
            """
            Set bounded D_E through its raw-logit parameterization, plus w_EE and h.
            """
            D = float(np.clip(estimate["D_E"], 0.100001, 4.999999))
            p = np.clip((D - 0.1) / 4.9, 1e-6, 1.0 - 1e-6)
            raw = float(np.log(p / (1.0 - p)))

            with torch.no_grad():
                raw_D.copy_(torch.tensor(raw, dtype=raw_D.dtype, device=raw_D.device))
                model.w_EE.copy_(
                    torch.tensor(
                        float(np.clip(estimate["w_EE"], 0.5, 6.0)),
                        dtype=model.w_EE.dtype,
                        device=model.w_EE.device,
                    )
                )
                model.h.copy_(
                    torch.tensor(float(estimate["h"]), dtype=model.h.dtype, device=model.h.device)
                )


        # --------------------------- synthetic sampling ---------------------------- #

        def make_state_pool(E, n_points, seed, early_fraction=0.50, early_seconds=5.0,
                            fps=30.0):
            """
            Fixed state-training pool with deliberate oversampling of the early transient.
            """
            rng = np.random.default_rng(seed)
            T, H, W = E.shape
            n_early = int(round(n_points * early_fraction))
            n_uniform = n_points - n_early

            early_T = max(1, min(T, int(round(early_seconds * fps)) + 1))

            ti_e = rng.integers(0, early_T, size=n_early)
            yi_e = rng.integers(0, H, size=n_early)
            xi_e = rng.integers(0, W, size=n_early)

            ti_u = rng.integers(0, T, size=n_uniform)
            yi_u = rng.integers(0, H, size=n_uniform)
            xi_u = rng.integers(0, W, size=n_uniform)

            ti = np.concatenate([ti_e, ti_u])
            yi = np.concatenate([yi_e, yi_u])
            xi = np.concatenate([xi_e, xi_u])

            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -1.0 + 2.0 * ti / float(T - 1)

            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            target = E[ti, yi, xi].astype(np.float32)[:, None]
            return xyt, target


        def make_ic_pool(E, n_points, seed):
            rng = np.random.default_rng(seed)
            _, H, W = E.shape
            total = H * W
            n = min(n_points, total)
            flat = rng.choice(total, n, replace=False)
            yi = flat // W
            xi = flat % W
            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -np.ones_like(x)
            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            target = E[0, yi, xi].astype(np.float32)[:, None]
            return xyt, target


        def make_derivative_pool(E, l_fov, t_seconds, n_points, seed,
                                 early_fraction=0.60, early_seconds=5.0):
            """
            Interior samples with centered finite-difference E_t and Laplacian targets.
            More weight is deliberately placed on the early transient.
            """
            rng = np.random.default_rng(seed)
            T, H, W = E.shape
            dt = float(np.median(np.diff(t_seconds)))
            dx = float(l_fov) / float(W - 1)
            dy = float(l_fov) / float(H - 1)

            early_last = min(T - 2, max(1, int(round(early_seconds / dt))))
            n_early = int(round(n_points * early_fraction))
            n_uniform = n_points - n_early

            ti_e = rng.integers(1, early_last + 1, size=n_early)
            yi_e = rng.integers(1, H - 1, size=n_early)
            xi_e = rng.integers(1, W - 1, size=n_early)

            ti_u = rng.integers(1, T - 1, size=n_uniform)
            yi_u = rng.integers(1, H - 1, size=n_uniform)
            xi_u = rng.integers(1, W - 1, size=n_uniform)

            ti = np.concatenate([ti_e, ti_u])
            yi = np.concatenate([yi_e, yi_u])
            xi = np.concatenate([xi_e, xi_u])

            E0 = E[ti, yi, xi].astype(np.float64)
            dE_dt = (
                E[ti + 1, yi, xi].astype(np.float64)
                - E[ti - 1, yi, xi].astype(np.float64)
            ) / (2.0 * dt)

            d2x = (
                E[ti, yi, xi + 1].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi, xi - 1].astype(np.float64)
            ) / (dx * dx)
            d2y = (
                E[ti, yi + 1, xi].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi - 1, xi].astype(np.float64)
            ) / (dy * dy)
            lap = d2x + d2y

            x = -1.0 + 2.0 * xi / float(W - 1)
            y = -1.0 + 2.0 * yi / float(H - 1)
            t = -1.0 + 2.0 * ti / float(T - 1)

            xyt = np.stack([x, y, t], axis=1).astype(np.float32)
            return (
                xyt,
                E0.astype(np.float32)[:, None],
                dE_dt.astype(np.float32)[:, None],
                lap.astype(np.float32)[:, None],
            )


        def batch_from_pool(pool_tensors, batch_size, generator=None):
            n = pool_tensors[0].shape[0]
            idx = torch.randint(
                0, n, (batch_size,),
                device=pool_tensors[0].device,
                generator=generator,
            )
            return tuple(x[idx] for x in pool_tensors)


        # ----------------------- autograd derivative functions --------------------- #

        def state_and_derivatives(model, xyt, duration, l_fov, create_graph=True):
            """
            Returns E, physical E_t, and physical Laplacian(E).
            xyt must be normalized to [-1,1]^3.
            """
            xyt = xyt.clone().detach().requires_grad_(True)
            E = model.forward_E(xyt)

            grad_E = torch.autograd.grad(
                E, xyt,
                grad_outputs=torch.ones_like(E),
                create_graph=True,
                retain_graph=True,
            )[0]

            dE_dx = grad_E[:, 0:1]
            dE_dy = grad_E[:, 1:2]
            dE_dt_norm = grad_E[:, 2:3]

            d2x = torch.autograd.grad(
                dE_dx, xyt,
                grad_outputs=torch.ones_like(dE_dx),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 0:1]

            d2y = torch.autograd.grad(
                dE_dy, xyt,
                grad_outputs=torch.ones_like(dE_dy),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 1:2]

            dE_dt = (2.0 / float(duration)) * dE_dt_norm
            lap = (4.0 / float(l_fov) ** 2) * (d2x + d2y)
            return E, dE_dt, lap


        def physics_residual(model, E, dE_dt, lap, tau):
            D = model.forward_D()
            w = torch.clamp(model.w_EE, 0.5, 6.0)
            h = model.h
            rhs = -E + torch.sigmoid(w * E + D * lap + h)
            return tau * dE_dt - rhs


        # ---------------------------- evaluation ---------------------------------- #

        @torch.no_grad()
        def predict_state(model, xyt, batch_size=4096):
            vals = []
            for s in range(0, len(xyt), batch_size):
                vals.append(
                    model.forward_E(xyt[s:s+batch_size]).detach().cpu().numpy().ravel()
                )
            return np.concatenate(vals)


        def predict_derivatives(model, xyt, duration, l_fov, batch_size=512):
            Es, Ts, Ls = [], [], []
            model.eval()
            for s in range(0, len(xyt), batch_size):
                E, Et, Lap = state_and_derivatives(
                    model, xyt[s:s+batch_size], duration, l_fov, create_graph=False
                )
                Es.append(E.detach().cpu().numpy().ravel())
                Ts.append(Et.detach().cpu().numpy().ravel())
                Ls.append(Lap.detach().cpu().numpy().ravel())
            return np.concatenate(Es), np.concatenate(Ts), np.concatenate(Ls)


        def scatter(x, y, xlabel, ylabel, title, path, max_points=12000):
            rng = np.random.default_rng(123)
            if len(x) > max_points:
                idx = rng.choice(len(x), max_points, replace=False)
                x, y = x[idx], y[idx]
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(x, y, s=5, alpha=0.25)
            lo = min(float(np.min(x)), float(np.min(y)))
            hi = max(float(np.max(x)), float(np.max(y)))
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)


        # ------------------------------- training --------------------------------- #

        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
            p.add_argument("--output_dir",
                           default="synthetic_outputs/oracle_derivative_control")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--fourier_seed", type=int, default=2718)

            p.add_argument("--state_pool", type=int, default=60000)
            p.add_argument("--deriv_pool", type=int, default=24000)
            p.add_argument("--ic_pool", type=int, default=8192)

            p.add_argument("--state_batch", type=int, default=4096)
            p.add_argument("--deriv_batch", type=int, default=512)
            p.add_argument("--ic_batch", type=int, default=2048)

            p.add_argument("--stageA_steps", type=int, default=2500)
            p.add_argument("--stageB_steps", type=int, default=2500)
            p.add_argument("--stageC_adam_steps", type=int, default=1800)
            p.add_argument("--stageD_steps", type=int, default=1200)

            p.add_argument("--lrA", type=float, default=1e-3)
            p.add_argument("--lrB", type=float, default=2e-4)
            p.add_argument("--lrC", type=float, default=1e-2)
            p.add_argument("--lrD_net", type=float, default=2e-5)
            p.add_argument("--lrD_phys", type=float, default=5e-4)

            p.add_argument("--duration", type=float, default=30.0)
            p.add_argument("--noise_sigma", type=float, default=0.0)
            p.add_argument("--smooth_sigma_t", type=float, default=1.0,
                           help="Gaussian smoothing sigma in frames.")
            p.add_argument("--smooth_sigma_space", type=float, default=1.0,
                           help="Gaussian smoothing sigma in pixels for y/x.")
            p.add_argument("--clip_observed", action="store_true",
                           help="Clip noisy observed field to [0,1].")
            args = p.parse_args()

            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            set_seed(args.seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            d = np.load(args.data, allow_pickle=False)
            E_full = d["E"].astype(np.float32)
            t_seconds = d["t_seconds"].astype(np.float64)
            truth = {
                "D_E": float(d["D_E_true"]),
                "w_EE": float(d["w_EE_true"]),
                "h": float(d["h_true"]),
            }
            tau = float(d["tau"])
            l_fov = float(d["l_fov"])

            # Create an observed field exactly as we would have observations in a real
            # inverse problem. For the first control noise_sigma=0; later we can repeat
            # with noise_sigma=0.01.
            rng_obs = np.random.default_rng(args.seed + 404)
            E_obs = E_full.astype(np.float64).copy()
            if args.noise_sigma > 0:
                E_obs = E_obs + args.noise_sigma * rng_obs.standard_normal(E_obs.shape)
            if args.clip_observed:
                E_obs = np.clip(E_obs, 0.0, 1.0)
            E_obs = E_obs.astype(np.float32)

            # Mild regularized state representation used ONLY to construct
            # real-compatible pseudo-derivative targets.
            E_smooth = gaussian_filter(
                E_obs.astype(np.float64),
                sigma=(args.smooth_sigma_t, args.smooth_sigma_space, args.smooth_sigma_space),
                mode="reflect",
            ).astype(np.float32)

            # State/IC losses use the observed field; pseudo-derivatives come from the
            # smoothed observed field.
            sx, sy = make_state_pool(
                E_obs, args.state_pool, args.seed + 1,
                early_fraction=0.50, early_seconds=5.0
            )
            ix, iy = make_ic_pool(E_obs, args.ic_pool, args.seed + 2)

            dx, dE, dEt, dLap = make_derivative_pool(
                E_smooth, l_fov, t_seconds, args.deriv_pool, args.seed + 3,
                early_fraction=0.60, early_seconds=5.0
            )

            # Same sample coordinates, but derivatives from the clean synthetic truth
            # are retained strictly for diagnostic evaluation.
            dx_true, dE_true, dEt_true, dLap_true = make_derivative_pool(
                E_full, l_fov, t_seconds, args.deriv_pool, args.seed + 3,
                early_fraction=0.60, early_seconds=5.0
            )
            if not np.allclose(dx, dx_true):
                raise RuntimeError("Pseudo and truth derivative pools do not share coordinates.")

            S = (
                torch.from_numpy(sx).to(device),
                torch.from_numpy(sy).to(device),
            )
            I = (
                torch.from_numpy(ix).to(device),
                torch.from_numpy(iy).to(device),
            )
            DPOOL = (
                torch.from_numpy(dx).to(device),
                torch.from_numpy(dE).to(device),
                torch.from_numpy(dEt).to(device),
                torch.from_numpy(dLap).to(device),
            )
            DPOOL_TRUE = (
                torch.from_numpy(dx_true).to(device),
                torch.from_numpy(dE_true).to(device),
                torch.from_numpy(dEt_true).to(device),
                torch.from_numpy(dLap_true).to(device),
            )

            pseudo_regression = bounded_logit_regression(
                dE, dEt, dLap, tau=tau, eps=1e-4
            )
            print("\n=== Smoothed-observation regression initialization ===", flush=True)
            print(json.dumps(pseudo_regression, indent=2), flush=True)

            model = gPINN_WilsonCowan(fourier_seed=args.fourier_seed).to(device)

            raw_D, raw_D_name = find_raw_D_parameter(model)
            phys_ids = {id(raw_D), id(model.w_EE), id(model.h)}
            network_params = [p for p in model.parameters() if id(p) not in phys_ids]
            physical_params = [raw_D, model.w_EE, model.h]

            history = []

            # ---------------- Stage A: state + IC pretraining ---------------- #
            print("\n=== Stage A: state + initial-condition pretraining ===", flush=True)
            for p_ in physical_params:
                p_.requires_grad_(False)
            for p_ in network_params:
                p_.requires_grad_(True)

            optA = torch.optim.Adam(network_params, lr=args.lrA)

            for step in range(1, args.stageA_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss = loss_data + 1.0 * loss_ic

                optA.zero_grad(set_to_none=True)
                loss.backward()
                optA.step()

                if step == 1 or step % 100 == 0 or step == args.stageA_steps:
                    print(
                        f"A {step:5d}/{args.stageA_steps} "
                        f"loss={loss.item():.4e} data={loss_data.item():.4e} "
                        f"ic={loss_ic.item():.4e}",
                        flush=True,
                    )
                    history.append(["A", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    "", "", ""])

            # -------------- Stage B: oracle derivative/Sobolev fit ----------- #
            print("\n=== Stage B: pseudo-derivative training from smoothed observations ===", flush=True)
            optB = torch.optim.Adam(network_params, lr=args.lrB)

            for step in range(1, args.stageB_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)
                xd, Ed_true, Et_true, Lap_true = batch_from_pool(
                    DPOOL, args.deriv_batch
                )

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)
                Ed_pred, Et_pred, Lap_pred = state_and_derivatives(
                    model, xd, args.duration, l_fov, create_graph=True
                )

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss_Ed = normalized_mse(Ed_pred, Ed_true)
                loss_Et = normalized_mse(Et_pred, Et_true)
                loss_Lap = normalized_mse(Lap_pred, Lap_true)

                loss = (
                    1.0 * loss_data
                    + 0.75 * loss_ic
                    + 0.50 * loss_Ed
                    + 1.00 * loss_Et
                    + 1.00 * loss_Lap
                )

                optB.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network_params, max_norm=10.0)
                optB.step()

                if step == 1 or step % 50 == 0 or step == args.stageB_steps:
                    print(
                        f"B {step:5d}/{args.stageB_steps} loss={loss.item():.4e} "
                        f"data={loss_data.item():.3e} ic={loss_ic.item():.3e} "
                        f"E={loss_Ed.item():.3e} Et={loss_Et.item():.3e} "
                        f"Lap={loss_Lap.item():.3e}",
                        flush=True,
                    )
                    history.append(["B", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    float(loss_Et.item()), float(loss_Lap.item()), ""])

            # Evaluate derivative quality before parameter fitting.
            eval_n = min(12000, DPOOL[0].shape[0])
            eval_idx = torch.arange(eval_n, device=device)
            eval_x = DPOOL[0][eval_idx]
            eval_E_true = DPOOL_TRUE[1][eval_idx].detach().cpu().numpy().ravel()
            eval_Et_true = DPOOL_TRUE[2][eval_idx].detach().cpu().numpy().ravel()
            eval_Lap_true = DPOOL_TRUE[3][eval_idx].detach().cpu().numpy().ravel()

            eval_E_pseudo = DPOOL[1][eval_idx].detach().cpu().numpy().ravel()
            eval_Et_pseudo = DPOOL[2][eval_idx].detach().cpu().numpy().ravel()
            eval_Lap_pseudo = DPOOL[3][eval_idx].detach().cpu().numpy().ravel()

            E_pre, Et_pre, Lap_pre = predict_derivatives(
                model, eval_x, args.duration, l_fov, batch_size=args.deriv_batch
            )
            pre_param_derivative_metrics = {
                "E": metric_block(eval_E_true, E_pre),
                "E_t": metric_block(eval_Et_true, Et_pre),
                "laplacian": metric_block(eval_Lap_true, Lap_pre),
            }
            pseudo_derivative_fidelity = {
                "E_smooth_vs_truth": metric_block(eval_E_true, eval_E_pseudo),
                "E_t_pseudo_vs_truth": metric_block(eval_Et_true, eval_Et_pseudo),
                "laplacian_pseudo_vs_truth": metric_block(eval_Lap_true, eval_Lap_pseudo),
            }
            print("\nPseudo-derivative fidelity against clean synthetic truth:", flush=True)
            print(json.dumps(pseudo_derivative_fidelity, indent=2), flush=True)
            print("\nLearned derivative metrics after Stage B against clean truth:", flush=True)
            print(json.dumps(pre_param_derivative_metrics, indent=2), flush=True)

            # Regression is used as an initialization, not as ground truth.
            regression_estimate = pseudo_regression["estimate"]
            set_physical_parameters(model, raw_D, regression_estimate)
            print("\nPhysical parameters initialized from smoothed-observation regression:",
                  json.dumps(physical_parameter_dict(model), indent=2), flush=True)

            # ---------------- Stage C: parameter-only optimization ------------ #
            print("\n=== Stage C: freeze network, optimize D_E / w_EE / h ===", flush=True)
            for p_ in network_params:
                p_.requires_grad_(False)
            for p_ in physical_params:
                p_.requires_grad_(True)

            # Cache learned E/derivatives on the derivative pool. No weight gradients now.
            cache_E, cache_Et, cache_Lap = [], [], []
            model.eval()
            for s in range(0, DPOOL[0].shape[0], args.deriv_batch):
                Ee, Ete, Lape = state_and_derivatives(
                    model, DPOOL[0][s:s+args.deriv_batch],
                    args.duration, l_fov, create_graph=False
                )
                cache_E.append(Ee.detach())
                cache_Et.append(Ete.detach())
                cache_Lap.append(Lape.detach())

            cache_E = torch.cat(cache_E, dim=0)
            cache_Et = torch.cat(cache_Et, dim=0)
            cache_Lap = torch.cat(cache_Lap, dim=0)

            optC = torch.optim.Adam(physical_params, lr=args.lrC)

            for step in range(1, args.stageC_adam_steps + 1):
                n = cache_E.shape[0]
                idx = torch.randint(0, n, (min(8192, n),), device=device)
                residual = physics_residual(
                    model,
                    cache_E[idx],
                    cache_Et[idx],
                    cache_Lap[idx],
                    tau,
                )
                loss_phys = torch.mean(residual ** 2)

                optC.zero_grad(set_to_none=True)
                loss_phys.backward()
                optC.step()

                # Mirror the hard w_EE constraint used by the inverse pipeline.
                with torch.no_grad():
                    model.w_EE.clamp_(0.5, 6.0)

                if step == 1 or step % 100 == 0 or step == args.stageC_adam_steps:
                    est = physical_parameter_dict(model)
                    print(
                        f"C {step:5d}/{args.stageC_adam_steps} "
                        f"physics={loss_phys.item():.4e} "
                        f"D={est['D_E']:.6f} w={est['w_EE']:.6f} h={est['h']:.6f}",
                        flush=True,
                    )
                    history.append(["C", step, float(loss_phys.item()),
                                    "", "", "", "", json.dumps(est)])

            stageC_estimate = physical_parameter_dict(model)
            print("Stage C estimate:", json.dumps(stageC_estimate, indent=2), flush=True)

            # ---------------- Stage D: joint low-LR fine tuning --------------- #
            print("\n=== Stage D: joint derivative-aware physics fine-tuning ===", flush=True)
            for p_ in network_params:
                p_.requires_grad_(True)
            for p_ in physical_params:
                p_.requires_grad_(True)

            optD = torch.optim.Adam([
                {"params": network_params, "lr": args.lrD_net},
                {"params": physical_params, "lr": args.lrD_phys},
            ])

            for step in range(1, args.stageD_steps + 1):
                xb, yb = batch_from_pool(S, args.state_batch)
                xic, yic = batch_from_pool(I, args.ic_batch)
                xd, Ed_true, Et_true, Lap_true = batch_from_pool(
                    DPOOL, args.deriv_batch
                )

                pred = model.forward_E(xb)
                pred_ic = model.forward_E(xic)
                Ed_pred, Et_pred, Lap_pred = state_and_derivatives(
                    model, xd, args.duration, l_fov, create_graph=True
                )
                residual = physics_residual(
                    model, Ed_pred, Et_pred, Lap_pred, tau
                )

                loss_data = normalized_mse(pred, yb)
                loss_ic = normalized_mse(pred_ic, yic)
                loss_Ed = normalized_mse(Ed_pred, Ed_true)
                loss_Et = normalized_mse(Et_pred, Et_true)
                loss_Lap = normalized_mse(Lap_pred, Lap_true)

                # Scale the PDE residual by a fixed characteristic scale so it contributes
                # without overwhelming the Sobolev supervision.
                loss_phys = torch.mean(residual ** 2) / 1e-3

                loss = (
                    1.0 * loss_data
                    + 0.50 * loss_ic
                    + 0.50 * loss_Ed
                    + 1.00 * loss_Et
                    + 1.00 * loss_Lap
                    + 0.25 * loss_phys
                )

                optD.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optD.step()

                with torch.no_grad():
                    model.w_EE.clamp_(0.5, 6.0)

                if step == 1 or step % 50 == 0 or step == args.stageD_steps:
                    est = physical_parameter_dict(model)
                    print(
                        f"D {step:5d}/{args.stageD_steps} loss={loss.item():.4e} "
                        f"Et={loss_Et.item():.3e} Lap={loss_Lap.item():.3e} "
                        f"phys={loss_phys.item():.3e} "
                        f"D={est['D_E']:.6f} w={est['w_EE']:.6f} h={est['h']:.6f}",
                        flush=True,
                    )
                    history.append(["D", step, float(loss.item()),
                                    float(loss_data.item()), float(loss_ic.item()),
                                    float(loss_Et.item()), float(loss_Lap.item()),
                                    json.dumps(est)])

            # --------------------------- final evaluation ---------------------- #
            model.eval()
            final_E, final_Et, final_Lap = predict_derivatives(
                model, eval_x, args.duration, l_fov, batch_size=args.deriv_batch
            )

            final_derivative_metrics = {
                "E": metric_block(eval_E_true, final_E),
                "E_t": metric_block(eval_Et_true, final_Et),
                "laplacian": metric_block(eval_Lap_true, final_Lap),
            }

            final_estimate = physical_parameter_dict(model)
            final_relerr = relative_errors(truth, final_estimate)

            # Residual on learned field.
            res_learned = (
                tau * final_Et
                - (
                    -final_E
                    + sigmoid_np(
                        final_estimate["w_EE"] * final_E
                        + final_estimate["D_E"] * final_Lap
                        + final_estimate["h"]
                    )
                )
            )

            report = {
                "experiment": "synthetic_pseudoderivative_regression_control",
                "important_note": (
                    "Training derivative targets are estimated from a mildly smoothed observed "
                    "field, not from clean truth. Clean derivatives are used only for synthetic "
                    "evaluation. This design is intended to be transferable to real calcium data."
                ),
                "seed": args.seed,
                "fourier_seed": args.fourier_seed,
                "device": str(device),
                "truth": truth,
                "initial_parameterization": {
                    "D_E": float(0.1 + 4.9 * 0.5),
                    "w_EE": 2.0,
                    "h": -1.0,
                },
                "raw_D_parameter_name": raw_D_name,
                "pseudo_derivative_fidelity_against_truth": pseudo_derivative_fidelity,
                "smoothed_observation_regression": pseudo_regression,
                "after_stageB_derivative_metrics_against_truth": pre_param_derivative_metrics,
                "stageC_parameter_only_estimate": stageC_estimate,
                "final_estimate": final_estimate,
                "final_relative_error_percent": final_relerr,
                "final_derivative_metrics": final_derivative_metrics,
                "final_learned_field_pde_residual_mse": float(np.mean(res_learned ** 2)),
                "settings": vars(args),
            }

            print("\n=== FINAL REPORT ===", flush=True)
            print(json.dumps(report, indent=2), flush=True)

            tag = f"seed-{args.seed}_pseudoderiv_regression"
            (out / f"{tag}.json").write_text(json.dumps(report, indent=2))

            torch.save({
                "model_state": model.state_dict(),
                "report": report,
            }, out / f"{tag}.pt")

            with open(out / "training_history.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "stage", "step", "loss", "data_loss", "ic_loss",
                    "Et_loss", "lap_loss", "parameter_estimate"
                ])
                writer.writerows(history)

            scatter(
                eval_E_true, final_E,
                "Synthetic truth E", "Pseudo-derivative gPINN E",
                "State accuracy after pseudo-derivative control",
                out / "scatter_E.png",
            )
            scatter(
                eval_Et_true, final_Et,
                "Finite-difference truth E_t", "Pseudo-derivative gPINN E_t",
                "Time-derivative accuracy after pseudo-derivative control",
                out / "scatter_Et.png",
            )
            scatter(
                eval_Lap_true, final_Lap,
                "Finite-difference truth Laplacian(E)",
                "Pseudo-derivative gPINN Laplacian(E)",
                "Laplacian accuracy after pseudo-derivative control",
                out / "scatter_laplacian.png",
            )

            print(f"\nSaved outputs to {out}", flush=True)
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# sweep: source sweep_pseudoderiv_regression.py
# ============================================================================
def sweep(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['sweep_pseudoderiv_regression.py'] + list(argv or [])
    try:
        """
        Noise × smoothing sensitivity sweep for real-compatible Wilson–Cowan
        pseudo-derivative regression.

        This experiment does NOT train a PINN. It asks whether the parameter estimator

            logit(tau E_t + E) = w_EE E + D_E Lap(E) + h

        remains accurate when the observed synthetic field is noisy and derivatives are
        estimated after Gaussian smoothing.

        Each Slurm array task corresponds to one (noise_sigma, noise_seed) condition and
        loops over a small grid of temporal/spatial smoothing strengths.

        Outputs:
          synthetic_outputs/regression_sensitivity/task-XXX.csv
          synthetic_outputs/regression_sensitivity/task-XXX.json
        """
        import argparse
        import csv
        import json
        from pathlib import Path

        import numpy as np
        from scipy.ndimage import gaussian_filter
        from scipy.optimize import lsq_linear


        NOISE_SEEDS = [42, 123, 456, 789, 2026]
        TASKS = [(0.0, 42)]
        for sigma in (0.005, 0.01, 0.02):
            TASKS.extend((sigma, seed) for seed in NOISE_SEEDS)

        SMOOTH_T = [0.0, 0.5, 1.0, 2.0]       # frames
        SMOOTH_XY = [0.0, 0.5, 1.0, 2.0]      # pixels


        def r2_score(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            ss_res = np.sum((a - b) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2)
            return float(1.0 - ss_res / (ss_tot + 1e-12))


        def rel_l2(y_true, y_pred):
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))


        def bounded_logit_regression(E, dE_dt, lap, tau, eps=1e-4):
            E = np.asarray(E).ravel().astype(np.float64)
            dE_dt = np.asarray(dE_dt).ravel().astype(np.float64)
            lap = np.asarray(lap).ravel().astype(np.float64)

            q = tau * dE_dt + E
            valid = np.isfinite(q) & np.isfinite(E) & np.isfinite(lap)
            valid &= (q > eps) & (q < 1.0 - eps)

            if valid.sum() < 100:
                return {
                    "success": False,
                    "n_valid": int(valid.sum()),
                    "fraction_valid": float(valid.mean()),
                }

            qv = q[valid]
            Ev = E[valid]
            Lv = lap[valid]

            y = np.log(qv / (1.0 - qv))
            X = np.column_stack([Ev, Lv, np.ones_like(Ev)])

            fit = lsq_linear(
                X, y,
                bounds=(
                    np.array([0.5, 0.1, -np.inf]),   # w, D, h
                    np.array([6.0, 5.0,  np.inf]),
                ),
                method="trf",
                lsmr_tol="auto",
                verbose=0,
            )

            w_hat, D_hat, h_hat = map(float, fit.x)

            Xstd = np.column_stack([
                (Ev - Ev.mean()) / (Ev.std() + 1e-12),
                (Lv - Lv.mean()) / (Lv.std() + 1e-12),
            ])

            return {
                "success": bool(fit.success),
                "n_valid": int(valid.sum()),
                "fraction_valid": float(valid.mean()),
                "D_E": D_hat,
                "w_EE": w_hat,
                "h": h_hat,
                "cost": float(fit.cost),
                "condition_number_raw_X": float(np.linalg.cond(X)),
                "condition_number_standardized_E_lap": float(np.linalg.cond(Xstd)),
                "corr_E_laplacian": float(np.corrcoef(Ev, Lv)[0, 1]),
            }


        def make_fixed_indices(shape, t_seconds, n_points, seed,
                               early_fraction=0.60, early_seconds=5.0):
            T, H, W = shape
            rng = np.random.default_rng(seed)
            dt = float(np.median(np.diff(t_seconds)))

            early_last = min(T - 2, max(1, int(round(early_seconds / dt))))
            n_early = int(round(n_points * early_fraction))
            n_uniform = n_points - n_early

            ti_e = rng.integers(1, early_last + 1, size=n_early)
            yi_e = rng.integers(1, H - 1, size=n_early)
            xi_e = rng.integers(1, W - 1, size=n_early)

            ti_u = rng.integers(1, T - 1, size=n_uniform)
            yi_u = rng.integers(1, H - 1, size=n_uniform)
            xi_u = rng.integers(1, W - 1, size=n_uniform)

            return (
                np.concatenate([ti_e, ti_u]),
                np.concatenate([yi_e, yi_u]),
                np.concatenate([xi_e, xi_u]),
            )


        def derivatives_at_indices(E, ti, yi, xi, l_fov, t_seconds):
            T, H, W = E.shape
            dt = float(np.median(np.diff(t_seconds)))
            dx = float(l_fov) / float(W - 1)
            dy = float(l_fov) / float(H - 1)

            E0 = E[ti, yi, xi].astype(np.float64)

            dE_dt = (
                E[ti + 1, yi, xi].astype(np.float64)
                - E[ti - 1, yi, xi].astype(np.float64)
            ) / (2.0 * dt)

            d2x = (
                E[ti, yi, xi + 1].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi, xi - 1].astype(np.float64)
            ) / (dx * dx)

            d2y = (
                E[ti, yi + 1, xi].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi - 1, xi].astype(np.float64)
            ) / (dy * dy)

            return E0, dE_dt, d2x + d2y


        def pct_err(est, true):
            return float(100.0 * abs(est - true) / (abs(true) + 1e-12))


        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
            p.add_argument("--output_dir",
                           default="synthetic_outputs/regression_sensitivity")
            p.add_argument("--task_id", type=int, required=True)
            p.add_argument("--n_points", type=int, default=24000)
            p.add_argument("--clip_observed", action="store_true")
            args = p.parse_args()

            if args.task_id < 0 or args.task_id >= len(TASKS):
                raise ValueError(f"task_id must be 0..{len(TASKS)-1}")

            noise_sigma, noise_seed = TASKS[args.task_id]

            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            d = np.load(args.data, allow_pickle=False)
            E_true = d["E"].astype(np.float32)
            t_seconds = d["t_seconds"].astype(np.float64)
            tau = float(d["tau"])
            l_fov = float(d["l_fov"])

            truth = {
                "D_E": float(d["D_E_true"]),
                "w_EE": float(d["w_EE_true"]),
                "h": float(d["h_true"]),
            }

            # Same evaluation coordinates for every smoothing condition in this task.
            ti, yi, xi = make_fixed_indices(
                E_true.shape, t_seconds, args.n_points, seed=909
            )

            E_truth_s, Et_truth_s, Lap_truth_s = derivatives_at_indices(
                E_true, ti, yi, xi, l_fov, t_seconds
            )

            rng = np.random.default_rng(noise_seed + 404)
            E_obs = E_true.astype(np.float64).copy()
            if noise_sigma > 0:
                E_obs += noise_sigma * rng.standard_normal(E_obs.shape)
            if args.clip_observed:
                E_obs = np.clip(E_obs, 0.0, 1.0)

            rows = []

            print(
                f"Task {args.task_id}: noise_sigma={noise_sigma}, "
                f"noise_seed={noise_seed}", flush=True
            )

            for sigma_t in SMOOTH_T:
                for sigma_xy in SMOOTH_XY:
                    if sigma_t == 0.0 and sigma_xy == 0.0:
                        E_smooth = E_obs
                    else:
                        E_smooth = gaussian_filter(
                            E_obs,
                            sigma=(sigma_t, sigma_xy, sigma_xy),
                            mode="reflect",
                        )

                    E_s, Et_s, Lap_s = derivatives_at_indices(
                        E_smooth, ti, yi, xi, l_fov, t_seconds
                    )

                    reg = bounded_logit_regression(E_s, Et_s, Lap_s, tau=tau)

                    row = {
                        "task_id": args.task_id,
                        "noise_sigma": noise_sigma,
                        "noise_seed": noise_seed,
                        "smooth_sigma_t_frames": sigma_t,
                        "smooth_sigma_space_pixels": sigma_xy,
                        "n_points": args.n_points,
                        "clip_observed": bool(args.clip_observed),
                        "E_smooth_r2_vs_truth": r2_score(E_truth_s, E_s),
                        "Et_pseudo_r2_vs_truth": r2_score(Et_truth_s, Et_s),
                        "Et_pseudo_rel_l2": rel_l2(Et_truth_s, Et_s),
                        "lap_pseudo_r2_vs_truth": r2_score(Lap_truth_s, Lap_s),
                        "lap_pseudo_rel_l2": rel_l2(Lap_truth_s, Lap_s),
                        "success": reg.get("success", False),
                        "n_valid": reg.get("n_valid", 0),
                        "fraction_valid": reg.get("fraction_valid", 0.0),
                    }

                    if reg.get("success", False):
                        row.update({
                            "D_E_true": truth["D_E"],
                            "D_E_est": reg["D_E"],
                            "D_E_relerr_pct": pct_err(reg["D_E"], truth["D_E"]),
                            "w_EE_true": truth["w_EE"],
                            "w_EE_est": reg["w_EE"],
                            "w_EE_relerr_pct": pct_err(reg["w_EE"], truth["w_EE"]),
                            "h_true": truth["h"],
                            "h_est": reg["h"],
                            "h_relerr_pct": pct_err(reg["h"], truth["h"]),
                            "mean_param_relerr_pct": float(np.mean([
                                pct_err(reg["D_E"], truth["D_E"]),
                                pct_err(reg["w_EE"], truth["w_EE"]),
                                pct_err(reg["h"], truth["h"]),
                            ])),
                            "max_param_relerr_pct": float(np.max([
                                pct_err(reg["D_E"], truth["D_E"]),
                                pct_err(reg["w_EE"], truth["w_EE"]),
                                pct_err(reg["h"], truth["h"]),
                            ])),
                            "condition_number_raw_X": reg["condition_number_raw_X"],
                            "condition_number_standardized_E_lap":
                                reg["condition_number_standardized_E_lap"],
                            "corr_E_laplacian": reg["corr_E_laplacian"],
                            "cost": reg["cost"],
                        })

                    rows.append(row)

                    msg = (
                        f"sigma_t={sigma_t:>3}, sigma_xy={sigma_xy:>3} | "
                        f"valid={row['fraction_valid']:.3f}"
                    )
                    if row["success"]:
                        msg += (
                            f" | D={row['D_E_est']:.4f} "
                            f"w={row['w_EE_est']:.4f} h={row['h_est']:.4f} "
                            f"| mean_err={row['mean_param_relerr_pct']:.2f}%"
                        )
                    print(msg, flush=True)

            csv_path = out / f"task-{args.task_id:03d}.csv"
            fieldnames = sorted({k for r in rows for k in r.keys()})
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            payload = {
                "task_id": args.task_id,
                "noise_sigma": noise_sigma,
                "noise_seed": noise_seed,
                "truth": truth,
                "smooth_sigma_t_values": SMOOTH_T,
                "smooth_sigma_space_values": SMOOTH_XY,
                "rows": rows,
            }
            json_path = out / f"task-{args.task_id:03d}.json"
            json_path.write_text(json.dumps(payload, indent=2))

            print(f"\nSaved {csv_path}", flush=True)
            print(f"Saved {json_path}", flush=True)
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# aggregate_sweep: source aggregate_pseudoderiv_regression_sweep.py
# ============================================================================
def aggregate_sweep(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['aggregate_pseudoderiv_regression_sweep.py'] + list(argv or [])
    try:
        """
        Aggregate noise × smoothing regression sensitivity sweep.
        Produces:
          all_runs.csv
          smoothing_summary.csv
          best_by_noise.csv
          robust_smoothing_ranking.csv
          sensitivity_summary.json
          heatmap_mean_error_noise-*.png
          heatmap_D_error_noise-*.png
        """
        import argparse
        import json
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt


        def save_heatmap(df, value_col, title, path):
            piv = df.pivot(
                index="smooth_sigma_t_frames",
                columns="smooth_sigma_space_pixels",
                values=value_col,
            ).sort_index().sort_index(axis=1)

            fig, ax = plt.subplots(figsize=(7, 5.5))
            im = ax.imshow(piv.values, origin="lower", aspect="auto")
            ax.set_xticks(np.arange(len(piv.columns)))
            ax.set_xticklabels([f"{x:g}" for x in piv.columns])
            ax.set_yticks(np.arange(len(piv.index)))
            ax.set_yticklabels([f"{x:g}" for x in piv.index])
            ax.set_xlabel("Spatial Gaussian sigma (pixels)")
            ax.set_ylabel("Temporal Gaussian sigma (frames)")
            ax.set_title(title)

            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    val = piv.iloc[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

            fig.colorbar(im, ax=ax, label="Relative error (%)")
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)


        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--input_dir",
                           default="synthetic_outputs/regression_sensitivity")
            args = p.parse_args()

            out = Path(args.input_dir)
            files = sorted(out.glob("task-*.csv"))
            if not files:
                raise FileNotFoundError(f"No task CSV files found in {out}")

            dfs = [pd.read_csv(f) for f in files]
            all_df = pd.concat(dfs, ignore_index=True)
            all_df.to_csv(out / "all_runs.csv", index=False)

            good = all_df[all_df["success"] == True].copy()
            if good.empty:
                raise RuntimeError("No successful regression rows.")

            group_cols = [
                "noise_sigma",
                "smooth_sigma_t_frames",
                "smooth_sigma_space_pixels",
            ]

            summary = good.groupby(group_cols).agg(
                n=("D_E_est", "count"),
                D_E_mean=("D_E_est", "mean"),
                D_E_sd=("D_E_est", "std"),
                D_E_relerr_mean_pct=("D_E_relerr_pct", "mean"),
                D_E_relerr_sd_pct=("D_E_relerr_pct", "std"),
                w_EE_mean=("w_EE_est", "mean"),
                w_EE_sd=("w_EE_est", "std"),
                w_EE_relerr_mean_pct=("w_EE_relerr_pct", "mean"),
                w_EE_relerr_sd_pct=("w_EE_relerr_pct", "std"),
                h_mean=("h_est", "mean"),
                h_sd=("h_est", "std"),
                h_relerr_mean_pct=("h_relerr_pct", "mean"),
                h_relerr_sd_pct=("h_relerr_pct", "std"),
                mean_param_relerr_mean_pct=("mean_param_relerr_pct", "mean"),
                max_param_relerr_mean_pct=("max_param_relerr_pct", "mean"),
                valid_fraction_mean=("fraction_valid", "mean"),
                cond_standardized_mean=("condition_number_standardized_E_lap", "mean"),
                Et_r2_mean=("Et_pseudo_r2_vs_truth", "mean"),
                lap_r2_mean=("lap_pseudo_r2_vs_truth", "mean"),
            ).reset_index()

            summary.to_csv(out / "smoothing_summary.csv", index=False)

            best_rows = []
            for noise, sub in summary.groupby("noise_sigma"):
                best = sub.sort_values(
                    ["mean_param_relerr_mean_pct", "max_param_relerr_mean_pct"]
                ).iloc[0]
                best_rows.append(best)
            best_by_noise = pd.DataFrame(best_rows)
            best_by_noise.to_csv(out / "best_by_noise.csv", index=False)

            # Robust smoothing choice: optimize across all nonzero noise levels and seeds.
            noisy = good[good["noise_sigma"] > 0].copy()
            robust = noisy.groupby(
                ["smooth_sigma_t_frames", "smooth_sigma_space_pixels"]
            ).agg(
                n=("D_E_est", "count"),
                mean_param_relerr_pct=("mean_param_relerr_pct", "mean"),
                max_param_relerr_mean_pct=("max_param_relerr_pct", "mean"),
                D_E_relerr_mean_pct=("D_E_relerr_pct", "mean"),
                w_EE_relerr_mean_pct=("w_EE_relerr_pct", "mean"),
                h_relerr_mean_pct=("h_relerr_pct", "mean"),
                valid_fraction_mean=("fraction_valid", "mean"),
                cond_standardized_mean=("condition_number_standardized_E_lap", "mean"),
            ).reset_index().sort_values(
                ["mean_param_relerr_pct", "max_param_relerr_mean_pct"]
            )
            robust.to_csv(out / "robust_smoothing_ranking.csv", index=False)

            # Heatmaps use mean across noise seeds for each noise level.
            for noise, sub in summary.groupby("noise_sigma"):
                tag = f"{noise:.3f}".replace(".", "p")
                save_heatmap(
                    sub,
                    "mean_param_relerr_mean_pct",
                    f"Mean parameter error, noise sigma={noise:g}",
                    out / f"heatmap_mean_error_noise-{tag}.png",
                )
                save_heatmap(
                    sub,
                    "D_E_relerr_mean_pct",
                    f"D_E relative error, noise sigma={noise:g}",
                    out / f"heatmap_D_error_noise-{tag}.png",
                )

            robust_best = robust.iloc[0].to_dict()
            payload = {
                "n_task_files": len(files),
                "n_rows": int(len(all_df)),
                "n_success": int(len(good)),
                "best_by_noise": best_by_noise.to_dict(orient="records"),
                "robust_best_nonzero_noise": robust_best,
                "interpretation_note": (
                    "The robust smoothing setting minimizes average parameter relative "
                    "error across sigma=0.005, 0.01, and 0.02 noise conditions. "
                    "Synthetic truth is used only to score sensitivity, not in regression."
                ),
            }
            (out / "sensitivity_summary.json").write_text(
                json.dumps(payload, indent=2, default=float)
            )

            print("\n=== BEST SMOOTHING BY NOISE LEVEL ===")
            cols = [
                "noise_sigma",
                "smooth_sigma_t_frames",
                "smooth_sigma_space_pixels",
                "D_E_relerr_mean_pct",
                "w_EE_relerr_mean_pct",
                "h_relerr_mean_pct",
                "mean_param_relerr_mean_pct",
                "valid_fraction_mean",
            ]
            print(best_by_noise[cols].to_string(index=False))

            print("\n=== ROBUST BEST ACROSS NONZERO NOISE LEVELS ===")
            print(robust.head(10).to_string(index=False))

            print(f"\nSaved outputs to {out}")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# selector: source synthetic_model_selection_control.py
# ============================================================================
def selector(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['synthetic_model_selection_control.py'] + list(argv or [])
    try:
        """Synthetic positive/null control for the final Wilson-Cowan model selector.

        Applies the *unchanged* regression-assisted model-selection logic from
        train_gpinn_narval_final.py to two independently generated synthetic trajectories:
          1) positive spatial coupling: D_E*=0.60
          2) null spatial coupling:     D_E*=0.00

        No neural-network training is performed. The purpose is to test whether the final
        selector can distinguish a supported Laplacian term from an absent one under the
        same decision criteria used for the real-data analysis.
        """
        import argparse
        import json
        from pathlib import Path
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd
        from scipy.ndimage import gaussian_filter

        from .training import (
            PRIMARY_SIGMA_XY_MM,
            compute_pseudo_fields,
            select_model,
            smoothing_D_estimates,
        )


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--positive", default="synthetic_data/wc_twin_truth.npz")
            p.add_argument("--null", default="synthetic_data/wc_twin_truth_D0.npz")
            p.add_argument("--out_dir", default="outputs/synthetic_model_selection_control")
            p.add_argument("--noise_sigmas", nargs="+", type=float, default=[0.0, 0.01])
            p.add_argument("--noise_seed", type=int, default=42)

            # Exact final selector defaults.
            p.add_argument("--nested_folds", type=int, default=5)
            p.add_argument("--bootstrap_reps", type=int, default=1000)
            p.add_argument("--bootstrap_block_frames", type=int, default=30)
            p.add_argument("--eps_logit", type=float, default=1e-4)
            p.add_argument("--min_valid_fraction", type=float, default=0.90)
            p.add_argument("--max_standardized_condition", type=float, default=10.0)
            p.add_argument("--min_logit_improvement_pct", type=float, default=0.10)
            p.add_argument("--min_pde_improvement_pct", type=float, default=0.10)
            p.add_argument("--min_positive_fold_fraction", type=float, default=0.60)
            p.add_argument("--max_smoothing_relative_range", type=float, default=1.00)
            p.add_argument("--w_min", type=float, default=0.5)
            p.add_argument("--w_max", type=float, default=6.0)
            p.add_argument("--seed", type=int, default=42)
            return p.parse_args()


        def load_case(path: Path):
            d = np.load(path, allow_pickle=False)
            return {
                "E": d["E"].astype(np.float64),
                "D_E_true": float(d["D_E_true"]),
                "w_EE_true": float(d["w_EE_true"]),
                "h_true": float(d["h_true"]),
                "tau": float(d["tau"]),
                "l_fov": float(d["l_fov"]),
                "fps": 1.0 / float(np.median(np.diff(d["t_seconds"].astype(np.float64)))),
            }


        def run_one(label: str, case: dict, noise_sigma: float, args):
            E = case["E"].copy()
            if noise_sigma > 0:
                rng = np.random.default_rng(args.noise_seed)
                E = np.clip(E + rng.normal(0.0, noise_sigma, size=E.shape), 0.0, 1.0)

            _, _, W = E.shape
            dx = case["l_fov"] / (W - 1)
            sigma_xy_px = PRIMARY_SIGMA_XY_MM / dx

            E_sm = gaussian_filter(
                E,
                sigma=(2.0, sigma_xy_px, sigma_xy_px),
                mode="reflect",
            )
            E0, Et, Lap, q, ylogit, valid = compute_pseudo_fields(
                E_sm,
                case["fps"],
                case["l_fov"],
                case["tau"],
                args.eps_logit,
            )

            sargs = SimpleNamespace(
                tau=case["tau"],
                nested_folds=args.nested_folds,
                bootstrap_block_frames=args.bootstrap_block_frames,
                bootstrap_reps=args.bootstrap_reps,
                seed=args.seed,
                w_min=args.w_min,
                w_max=args.w_max,
            )
            select = select_model(E0, Et, Lap, ylogit, valid, sargs, chunk_idx=0)
            smooth_results, smooth_rel_range, smooth_all_positive = smoothing_D_estimates(
                E,
                case["fps"],
                case["l_fov"],
                case["tau"],
                args.eps_logit,
            )

            b_full = select["full_beta"]
            b_red = select["reduced_beta"]
            criteria = {
                "valid_fraction": select["valid_fraction"] >= args.min_valid_fraction,
                "conditioning": select["condition_standardized"] <= args.max_standardized_condition,
                "D_point_positive": bool(float(b_full[1]) > 0.0),
                "D_CI95_positive": bool(select["ci_excludes_zero_positive"]),
                "logit_mean_improvement": (
                    select["mean_logit_improvement_pct"] >= args.min_logit_improvement_pct
                ),
                "pde_mean_improvement": (
                    select["mean_pde_improvement_pct"] >= args.min_pde_improvement_pct
                ),
                "logit_fold_consistency": (
                    select["fraction_logit_folds_positive"] >= args.min_positive_fold_fraction
                ),
                "pde_fold_consistency": (
                    select["fraction_pde_folds_positive"] >= args.min_positive_fold_fraction
                ),
                "smoothing_all_positive": bool(smooth_all_positive),
                "smoothing_relative_range": (
                    smooth_rel_range <= args.max_smoothing_relative_range
                ),
                "w_in_bounds": bool(select["w_in_bounds"]),
            }

            use_full = bool(all(criteria.values()))
            selected_model = "full" if use_full else "reduced"
            expected_model = "full" if case["D_E_true"] > 0 else "reduced"

            return {
                "case": label,
                "noise_sigma": float(noise_sigma),
                "noise_seed": int(args.noise_seed),
                "truth_D_E": case["D_E_true"],
                "truth_w_EE": case["w_EE_true"],
                "truth_h": case["h_true"],
                "expected_model": expected_model,
                "selected_model": selected_model,
                "selection_correct": bool(selected_model == expected_model),
                "full_w_EE": float(b_full[0]),
                "full_D_E": float(b_full[1]),
                "full_h": float(b_full[2]),
                "reduced_w_EE": float(b_red[0]),
                "reduced_h": float(b_red[1]),
                "D_ci95_low": float(select["bootstrap"]["ci95_low"]),
                "D_ci95_high": float(select["bootstrap"]["ci95_high"]),
                "valid_fraction": float(select["valid_fraction"]),
                "condition_standardized": float(select["condition_standardized"]),
                "mean_logit_improvement_pct": float(select["mean_logit_improvement_pct"]),
                "mean_pde_improvement_pct": float(select["mean_pde_improvement_pct"]),
                "fraction_logit_folds_positive": float(select["fraction_logit_folds_positive"]),
                "fraction_pde_folds_positive": float(select["fraction_pde_folds_positive"]),
                "smoothing_D_half": float(smooth_results["half"]["D"]),
                "smoothing_D_primary": float(smooth_results["primary"]["D"]),
                "smoothing_D_double": float(smooth_results["double"]["D"]),
                "smoothing_relative_range": float(smooth_rel_range),
                **{f"criterion_{k}": bool(v) for k, v in criteria.items()},
            }


        def main():
            args = parse_args()
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)

            cases = {
                "positive_D0p60": load_case(Path(args.positive)),
                "null_D0": load_case(Path(args.null)),
            }

            # Guard against accidentally auditing the wrong files.
            if not np.isclose(cases["positive_D0p60"]["D_E_true"], 0.60):
                raise ValueError("Positive-control file does not have D_E_true=0.60")
            if not np.isclose(cases["null_D0"]["D_E_true"], 0.0):
                raise ValueError("Null-control file does not have D_E_true=0.0")

            rows = []
            for label, case in cases.items():
                for noise in args.noise_sigmas:
                    print(f"\n=== {label} | noise={noise:g} ===", flush=True)
                    row = run_one(label, case, float(noise), args)
                    rows.append(row)
                    print(json.dumps(row, indent=2), flush=True)

            df = pd.DataFrame(rows)
            df.to_csv(out / "synthetic_model_selection_control.csv", index=False)

            summary = {
                "purpose": (
                    "Positive/null synthetic control for the unchanged final model selector. "
                    "A discriminating selector should choose full for D_E*=0.60 and reduced for D_E*=0."
                ),
                "thresholds": {
                    "min_valid_fraction": args.min_valid_fraction,
                    "max_standardized_condition": args.max_standardized_condition,
                    "min_logit_improvement_pct": args.min_logit_improvement_pct,
                    "min_pde_improvement_pct": args.min_pde_improvement_pct,
                    "min_positive_fold_fraction": args.min_positive_fold_fraction,
                    "max_smoothing_relative_range": args.max_smoothing_relative_range,
                    "w_bounds": [args.w_min, args.w_max],
                    "bootstrap_reps": args.bootstrap_reps,
                    "bootstrap_block_frames": args.bootstrap_block_frames,
                },
                "n_runs": int(len(df)),
                "n_correct": int(df["selection_correct"].sum()),
                "all_correct": bool(df["selection_correct"].all()),
                "runs": rows,
            }
            (out / "synthetic_model_selection_control.json").write_text(
                json.dumps(summary, indent=2)
            )

            print("\n=== COMPACT SUMMARY ===")
            print(df[[
                "case", "noise_sigma", "truth_D_E", "expected_model", "selected_model",
                "selection_correct", "full_D_E", "D_ci95_low", "D_ci95_high",
                "mean_logit_improvement_pct", "mean_pde_improvement_pct",
            ]].to_string(index=False))
            print(f"\nSaved to {out}")
        return main()
    finally:
        _sys.argv = _old_argv
