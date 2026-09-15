"""Publication-data extraction, tables, and figures.

Each command below is a namespaced consolidation of a script from the validated
reproducibility archive. The numerical implementation is preserved while the
public repository exposes a compact command surface.
"""
from __future__ import annotations

import sys as _sys

# ============================================================================
# extract: source extract_publication_results_v2.py
# ============================================================================
def extract(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['extract_publication_results_v2.py'] + list(argv or [])
    try:
        """
        Extract the final Wilson-Cowan/gPINN manuscript results for publication V2
        from the completed Narval experiments into one reproducible publication-data directory.

        This script does NOT retrain anything. It only reads completed outputs.
        It produces the manuscript tables plus master CSV/JSON files used by the
        publication plotting script.
        """
        import argparse
        import json
        import math
        from pathlib import Path

        import numpy as np
        import pandas as pd


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--project_root", default=".")
            p.add_argument("--final_dir", default="outputs/final_dual_model_sequential20_cont005")
            p.add_argument("--no_cont_dir", default="outputs/final_dual_model_sequential20")
            p.add_argument("--continuity_dir", default="outputs/continuity_ablation")
            p.add_argument("--synthetic_dir", default="synthetic_outputs")
            p.add_argument("--output_dir", default="outputs/publication_v2")
            return p.parse_args()


        def read_json(path: Path):
            return json.loads(path.read_text())


        def first_existing(paths):
            for p in paths:
                if p and Path(p).exists():
                    return Path(p)
            return None


        def recursive_first(root: Path, pattern: str):
            if not root.exists():
                return None
            hits = sorted(root.rglob(pattern))
            return hits[0] if hits else None


        def mean_sd(x):
            a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
            if len(a) == 0:
                return np.nan, np.nan
            return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


        def fmt_pm(mean, sd, digits=3):
            if not np.isfinite(mean):
                return "NA"
            return f"{mean:.{digits}f} ± {sd:.{digits}f}"


        def relerr(est, truth):
            return abs(float(est) - float(truth)) / max(abs(float(truth)), 1e-12) * 100.0


        def fmt_pct(x):
            """Compact manuscript display for relative errors."""
            x = float(x)
            if not np.isfinite(x):
                return "NA"
            return f"{x:.2f}" if abs(x) < 1 else f"{x:.1f}"


        def flatten_dict(d, prefix=""):
            out = {}
            for k, v in d.items():
                kk = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, dict):
                    out.update(flatten_dict(v, kk))
                else:
                    out[kk] = v
            return out


        def main():
            args = parse_args()
            root = Path(args.project_root).resolve()
            final_dir = root / args.final_dir
            no_cont_dir = root / args.no_cont_dir
            cont_dir = root / args.continuity_dir
            syn_dir = root / args.synthetic_dir
            out = root / args.output_dir
            tables = out / "tables"
            master = out / "master"
            tables.mkdir(parents=True, exist_ok=True)
            master.mkdir(parents=True, exist_ok=True)

            final_csv = first_existing([
                final_dir / "sequential20_continuity_results.csv",
                final_dir / "final_pipeline_results.csv",
            ])
            final_summary_path = first_existing([
                final_dir / "sequential20_continuity_summary.json",
                final_dir / "final_pipeline_summary.json",
            ])
            if final_csv is None or final_summary_path is None:
                raise FileNotFoundError(
                    f"Final run not found in {final_dir}. Expected continuity results CSV and summary JSON."
                )

            final = pd.read_csv(final_csv)
            final_summary = read_json(final_summary_path)
            if len(final) != 20:
                raise RuntimeError(f"Expected 20 final chunks, found {len(final)}")

            no_cont_csv = first_existing([no_cont_dir / "sequential20_results.csv", no_cont_dir / "final_pipeline_results.csv"])
            no_cont = pd.read_csv(no_cont_csv) if no_cont_csv else pd.DataFrame()
            no_cont_summary_path = first_existing([no_cont_dir / "sequential20_summary.json", no_cont_dir / "final_pipeline_summary.json"])
            no_cont_summary = read_json(no_cont_summary_path) if no_cont_summary_path else {}

            cont_runs_path = first_existing([cont_dir / "continuity_ablation_runs.csv"])
            cont_summary_csv_path = first_existing([cont_dir / "continuity_ablation_summary.csv"])
            cont_runs = pd.read_csv(cont_runs_path) if cont_runs_path else pd.DataFrame()
            cont_summary = pd.read_csv(cont_summary_csv_path) if cont_summary_csv_path else pd.DataFrame()

            # ------------------------------------------------------------------
            # Main Table 1: final pipeline and hyperparameters
            # ------------------------------------------------------------------
            table1_rows = [
                ("Data window", "30 s non-overlapping chunks"),
                ("Number of chunks", "20 (10 min session)"),
                ("Imaging grid", "64 × 64 after preprocessing/downsampling"),
                ("Sampling rate", "30 Hz"),
                ("Field of view", "10 mm"),
                ("Grid spacing", "10/(64−1) = 0.15873 mm/pixel"),
                ("State scaling", "Session-global min–max scaling (offline analysis)"),
                ("Pseudo-derivative smoothing", "σ_t = 2 frames; σ_xy = 0.15748 mm ≈ 0.992 px at 64×64"),
                ("Candidate physical models", "Full: w_EE, D_E, h; Reduced: D_E = 0"),
                ("Parameter estimator", "Finite-difference pseudo-derivative logit regression"),
                ("D_E uncertainty", "Temporal block bootstrap; 1000 replicates; 1 s blocks"),
                ("Nested model comparison", "5-fold random-pointwise cross-validation"),
                ("Minimum valid fraction", "0.90"),
                ("Maximum standardized condition number", "10"),
                ("Minimum full-model logit improvement", "0.10%"),
                ("Minimum full-model PDE improvement", "0.10%"),
                ("Minimum positive-fold fraction", "0.60"),
                ("Maximum smoothing relative range", "1.0"),
                ("w_EE admissible range", "[0.5, 6.0]"),
                ("Fourier feature dimension", "128 total random projections (64 spatial + 64 temporal); 256 after sin/cos concatenation"),
                ("Fourier scales (space/time)", "25 / 50"),
                ("State network", "6 hidden layers × 256 tanh units"),
                ("State train/validation points", "60,000 / 15,000 random pointwise"),
                ("Collocation points", "60,000"),
                ("Pseudo-derivative training points", "20,000"),
                ("Stage A", "2000 Adam steps, state-only pretraining"),
                ("Stage B", "3000 Adam steps, fixed-physics + derivative supervision"),
                ("L-BFGS polish", "150 iterations"),
                ("Physics loss weight", "0.05"),
                ("Pseudo-derivative loss weight", "0.001"),
                ("Gradient-residual weight μ_grad", "1×10⁻⁵"),
                ("Continuity weight λ_cont", "0.05 from chunk 1 onward"),
                ("Sequential information carried", "Neural state/Fourier representation only"),
                ("Not carried between chunks", "Physical parameters and optimizer state"),
                ("Validation interpretation", "Random-pointwise spatiotemporal interpolation, not forecasting"),
            ]
            table1 = pd.DataFrame(table1_rows, columns=["Component", "Final setting"])
            table1.to_csv(tables / "table1_final_pipeline_hyperparameters.csv", index=False)

            # ------------------------------------------------------------------
            # Main Table 2: synthetic recoverability/diagnostic evidence
            # ------------------------------------------------------------------
            syn_rows = []
            syn_recovery = first_existing([
                syn_dir / "synthetic_recovery_summary.csv",
                recursive_first(syn_dir, "synthetic_recovery_summary.csv"),
            ])
            if syn_recovery:
                s = pd.read_csv(syn_recovery)
                for _, r in s.iterrows():
                    noise = float(r["noise_sigma"])
                    syn_rows.append({
                        "Method": f"Baseline joint gPINN ({'clean' if noise == 0 else f'noise σ={noise:g}'})",
                        "n": int(r["n"]),
                        "D_E estimate": fmt_pm(float(r["D_E_mean"]), float(r["D_E_sd"]), 3),
                        "D_E rel. error (%)": fmt_pct(r["D_E_mean_relerr_pct"]),
                        "w_EE estimate": fmt_pm(float(r["w_EE_mean"]), float(r["w_EE_sd"]), 3),
                        "w_EE rel. error (%)": fmt_pct(r["w_EE_mean_relerr_pct"]),
                        "h estimate": fmt_pm(float(r["h_mean"]), float(r["h_sd"]), 3),
                        "h rel. error (%)": fmt_pct(r["h_mean_relerr_pct"]),
                        "Purpose": "Direct joint inverse gPINN baseline",
                    })

            # Oracle derivative control
            oracle_json = recursive_first(syn_dir, "*oracle_control.json")
            if oracle_json:
                d = read_json(oracle_json)
                truth = d.get("truth", {})
                est = d.get("final_estimate", {})
                syn_rows.append({
                    "Method": "Oracle derivative-supervised gPINN",
                    "n": 1,
                    "D_E estimate": f"{est.get('D_E', np.nan):.3f}",
                    "D_E rel. error (%)": fmt_pct(relerr(est.get("D_E", np.nan), truth.get("D_E", np.nan))),
                    "w_EE estimate": f"{est.get('w_EE', np.nan):.3f}",
                    "w_EE rel. error (%)": fmt_pct(relerr(est.get("w_EE", np.nan), truth.get("w_EE", np.nan))),
                    "h estimate": f"{est.get('h', np.nan):.3f}",
                    "h rel. error (%)": fmt_pct(relerr(est.get("h", np.nan), truth.get("h", np.nan))),
                    "Purpose": "Oracle diagnostic: asks whether better derivatives improve recovery",
                })

            staged_json = recursive_first(syn_dir, "*staged_nonoracle.json")
            if staged_json:
                d = read_json(staged_json)
                truth = d.get("truth", {})
                est = d.get("final_estimate", {})
                syn_rows.append({
                    "Method": "Staged non-oracle gPINN",
                    "n": 1,
                    "D_E estimate": f"{est.get('D_E', np.nan):.3f}",
                    "D_E rel. error (%)": fmt_pct(relerr(est.get("D_E", np.nan), truth.get("D_E", np.nan))),
                    "w_EE estimate": f"{est.get('w_EE', np.nan):.3f}",
                    "w_EE rel. error (%)": fmt_pct(relerr(est.get("w_EE", np.nan), truth.get("w_EE", np.nan))),
                    "h estimate": f"{est.get('h', np.nan):.3f}",
                    "h rel. error (%)": fmt_pct(relerr(est.get("h", np.nan), truth.get("h", np.nan))),
                    "Purpose": "Optimization-staging diagnostic",
                })

            pseudo_json = recursive_first(syn_dir, "*pseudoderiv_regression.json")
            if pseudo_json:
                d = read_json(pseudo_json)
                truth = d.get("truth", {})
                preg = d.get("smoothed_observation_regression", {}).get("estimate", {})
                if preg:
                    syn_rows.append({
                        "Method": "Pseudo-derivative regression",
                        "n": 1,
                        "D_E estimate": f"{preg.get('D_E', np.nan):.3f}",
                        "D_E rel. error (%)": fmt_pct(relerr(preg.get("D_E", np.nan), truth.get("D_E", np.nan))),
                        "w_EE estimate": f"{preg.get('w_EE', np.nan):.3f}",
                        "w_EE rel. error (%)": fmt_pct(relerr(preg.get("w_EE", np.nan), truth.get("w_EE", np.nan))),
                        "h estimate": f"{preg.get('h', np.nan):.3f}",
                        "h rel. error (%)": fmt_pct(relerr(preg.get("h", np.nan), truth.get("h", np.nan))),
                        "Purpose": "Final real-compatible parameter estimator",
                    })

            table2 = pd.DataFrame(syn_rows)
            if not table2.empty:
                table2.to_csv(tables / "table2_synthetic_recoverability.csv", index=False)

            # Numeric method-comparison file for Figure 2C.
            method_rows = []
            # Representative original joint gPINN checkpoint from the derivative diagnostic is
            # added below once the diagnostic JSON is loaded. Other controls are collected here.
            if oracle_json:
                d = read_json(oracle_json)
                est = d.get("final_estimate", {})
                method_rows.append({
                    "Method": "Oracle derivative-supervised gPINN",
                    "D_E": est.get("D_E", np.nan),
                    "w_EE": est.get("w_EE", np.nan),
                    "h": est.get("h", np.nan),
                })
            if staged_json:
                d = read_json(staged_json)
                est = d.get("final_estimate", {})
                method_rows.append({
                    "Method": "Staged non-oracle gPINN",
                    "D_E": est.get("D_E", np.nan),
                    "w_EE": est.get("w_EE", np.nan),
                    "h": est.get("h", np.nan),
                })
            if pseudo_json:
                d = read_json(pseudo_json)
                preg = d.get("smoothed_observation_regression", {}).get("estimate", {})
                if preg:
                    method_rows.append({
                        "Method": "Pseudo-derivative regression",
                        "D_E": preg.get("D_E", np.nan),
                        "w_EE": preg.get("w_EE", np.nan),
                        "h": preg.get("h", np.nan),
                    })

            # Derivative diagnostic is kept as a master evidence file for Figure 2.
            diag_json = recursive_first(syn_dir, "*_derivative_diagnostic.json")
            diagnostic = read_json(diag_json) if diag_json else {}
            if diagnostic:
                (master / "synthetic_derivative_diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
                chk = diagnostic.get("checkpoint_estimate", {})
                method_rows.insert(0, {
                    "Method": "Baseline joint gPINN",
                    "D_E": chk.get("D_E", np.nan),
                    "w_EE": chk.get("w_EE", np.nan),
                    "h": chk.get("h", np.nan),
                })
                truth = diagnostic.get("truth", {})
                method_rows.append({
                    "Method": "Truth",
                    "D_E": truth.get("D_E", np.nan),
                    "w_EE": truth.get("w_EE", np.nan),
                    "h": truth.get("h", np.nan),
                })
                pd.DataFrame(method_rows).to_csv(master / "synthetic_method_comparison.csv", index=False)

            sensitivity_json = recursive_first(syn_dir, "sensitivity_summary.json")
            sensitivity = read_json(sensitivity_json) if sensitivity_json else {}
            if sensitivity:
                (master / "synthetic_smoothing_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
            smoothing_summary = recursive_first(syn_dir, "smoothing_summary.csv")
            if smoothing_summary:
                pd.read_csv(smoothing_summary).to_csv(master / "synthetic_smoothing_summary.csv", index=False)

            # Copy synthetic run-level recovery if available for Figure 2.
            syn_runs = first_existing([
                syn_dir / "synthetic_recovery_runs.csv",
                recursive_first(syn_dir, "synthetic_recovery_runs.csv"),
            ])
            if syn_runs:
                pd.read_csv(syn_runs).to_csv(master / "synthetic_recovery_runs.csv", index=False)

            # ------------------------------------------------------------------
            # Main Table 3: final real-session evidence
            # ------------------------------------------------------------------
            w_mean, w_sd = mean_sd(final["selected_w_EE"])
            h_mean, h_sd = mean_sd(final["selected_h"])
            D_mean, D_sd = mean_sd(final["full_regression_D_E"])
            r2_mean, r2_sd = mean_sd(final["random_val_state_r2"])
            mse_mean, mse_sd = mean_sd(final["random_val_state_mse"])
            pde_mean, pde_sd = mean_sd(final["random_val_neural_pde_rmse"])

            ci_cross = ((final["D_boot_ci95_low"] <= 0) & (final["D_boot_ci95_high"] >= 0))
            n_red = int((final["selected_model"].astype(str).str.lower() == "reduced").sum())
            n_full = len(final) - n_red

            final_boundary = pd.to_numeric(final["boundary_pred_to_observed_ratio"], errors="coerce").dropna()
            final_bmean = float(final_boundary.mean())
            final_bsd = float(final_boundary.std(ddof=1))

            no_cont_r2_mean = np.nan
            no_cont_pde_mean = np.nan
            no_cont_bmean = np.nan
            if not no_cont.empty:
                no_cont_r2_mean = float(pd.to_numeric(no_cont["random_val_state_r2"], errors="coerce").mean())
                no_cont_pde_mean = float(pd.to_numeric(no_cont["random_val_neural_pde_rmse"], errors="coerce").mean())
                no_cont_bmean = float(pd.to_numeric(no_cont["boundary_pred_to_observed_ratio"], errors="coerce").dropna().mean())

            table3_rows = [
                ("Selected model", f"Reduced D_E=0 in {n_red}/20 chunks; full in {n_full}/20"),
                ("Unrestricted D_E", fmt_pm(D_mean, D_sd, 4)),
                ("D_E bootstrap intervals containing zero", f"{int(ci_cross.sum())}/20"),
                ("w_EE", fmt_pm(w_mean, w_sd, 3)),
                ("h", fmt_pm(h_mean, h_sd, 3)),
                ("Random-pointwise validation R²", fmt_pm(r2_mean, r2_sd, 4)),
                ("Random-pointwise validation MSE", fmt_pm(mse_mean, mse_sd, 6)),
                ("Neural PDE RMSE", fmt_pm(pde_mean, pde_sd, 4)),
                ("Boundary ratio, final λ_cont=0.05", fmt_pm(final_bmean, final_bsd, 3)),
                ("Boundary ratio, no continuity", f"{no_cont_bmean:.3f}" if np.isfinite(no_cont_bmean) else "NA"),
                ("Δ mean R², continuity − no continuity", f"{r2_mean - no_cont_r2_mean:+.4f}" if np.isfinite(no_cont_r2_mean) else "NA"),
                ("Δ mean PDE RMSE, continuity − no continuity", f"{pde_mean - no_cont_pde_mean:+.4f}" if np.isfinite(no_cont_pde_mean) else "NA"),
            ]
            table3 = pd.DataFrame(table3_rows, columns=["Outcome", "Final result"])
            table3.to_csv(tables / "table3_real_session_summary.csv", index=False)

            # ------------------------------------------------------------------
            # Supplementary tables
            # ------------------------------------------------------------------
            keep_cols = [
                "chunk_idx", "time_start_min", "selected_model", "selected_w_EE", "selected_D_E", "selected_h",
                "full_regression_D_E", "D_boot_ci95_low", "D_boot_ci95_high", "nested_logit_improvement_pct",
                "nested_pde_improvement_pct", "condition_standardized", "random_val_state_r2", "random_val_state_mse",
                "random_val_state_r", "random_val_Et_rmse_vs_pseudo", "random_val_Lap_rmse_vs_pseudo",
                "random_val_neural_pde_rmse", "boundary_pred_to_observed_ratio",
            ]
            keep_cols = [c for c in keep_cols if c in final.columns]
            final[keep_cols].to_csv(tables / "supp_table_s1_chunkwise_final_results.csv", index=False)

            criterion_cols = [c for c in final.columns if c.startswith("criterion_")]
            crit_out = final[["chunk_idx", "selected_model"] + criterion_cols].copy()
            crit_out.to_csv(tables / "supp_table_s2_model_selection_criteria.csv", index=False)

            if not cont_summary.empty:
                cont_summary.to_csv(tables / "supp_table_s3_continuity_ablation_summary.csv", index=False)
            if not cont_runs.empty:
                cont_runs.to_csv(master / "continuity_ablation_runs.csv", index=False)

            if not no_cont.empty:
                matched_cols = [
                    "chunk_idx", "random_val_state_r2", "random_val_state_mse", "random_val_neural_pde_rmse",
                    "pretrain_random_val_state_r2", "sequential_minus_independent_r2",
                    "independent_ref_state_r2", "independent_ref_state_mse", "independent_ref_neural_pde_rmse",
                ]
                matched_cols = [c for c in matched_cols if c in no_cont.columns]
                matched = no_cont[matched_cols].copy()
                if "sequential_minus_independent_r2" in matched.columns:
                    matched = matched[matched["sequential_minus_independent_r2"].notna()]
                matched.to_csv(tables / "supp_table_s4_sequential_vs_independent.csv", index=False)

            # Master real CSVs used by plotting script.
            final.to_csv(master / "final_continuity_chunkwise.csv", index=False)
            if not no_cont.empty:
                no_cont.to_csv(master / "no_continuity_chunkwise.csv", index=False)
            if not cont_summary.empty:
                cont_summary.to_csv(master / "continuity_ablation_summary.csv", index=False)

            # Machine-readable publication summary.
            publication_summary = {
                "final_run_source": str(final_csv),
                "n_chunks": int(len(final)),
                "model_selection": {"reduced": n_red, "full": n_full, "D_ci_crosses_zero": int(ci_cross.sum())},
                "real_parameters": {
                    "w_EE_mean": w_mean, "w_EE_sd": w_sd,
                    "h_mean": h_mean, "h_sd": h_sd,
                    "unrestricted_D_E_mean": D_mean, "unrestricted_D_E_sd": D_sd,
                },
                "final_performance": {
                    "random_val_r2_mean": r2_mean, "random_val_r2_sd": r2_sd,
                    "random_val_mse_mean": mse_mean, "random_val_mse_sd": mse_sd,
                    "neural_pde_rmse_mean": pde_mean, "neural_pde_rmse_sd": pde_sd,
                    "boundary_ratio_mean": final_bmean, "boundary_ratio_sd": final_bsd,
                },
                "no_continuity_reference": {
                    "random_val_r2_mean": no_cont_r2_mean,
                    "neural_pde_rmse_mean": no_cont_pde_mean,
                    "boundary_ratio_mean": no_cont_bmean,
                },
                "interpretation": {
                    "validation": "random-pointwise spatiotemporal interpolation, not forecasting",
                    "D_E": "effective Laplacian spatial-coupling coefficient; no stable support for inclusion in this resting-state session",
                },
                "available_optional_synthetic_files": {
                    "recovery_summary": str(syn_recovery) if syn_recovery else None,
                    "derivative_diagnostic": str(diag_json) if diag_json else None,
                    "sensitivity_summary": str(sensitivity_json) if sensitivity_json else None,
                },
            }
            (master / "publication_summary.json").write_text(json.dumps(publication_summary, indent=2))

            print("Publication extraction complete.")
            print(f"Main tables: {tables}")
            print(f"Master data: {master}")
            print(f"Final model selection: reduced {n_red}/20, full {n_full}/20")
            print(f"Final mean R2: {r2_mean:.6f}")
            print(f"Final mean boundary ratio: {final_bmean:.6f}")
            if not table2.empty:
                print(f"Synthetic Table 2 rows: {len(table2)}")
            else:
                print("WARNING: synthetic outputs were not found under synthetic_outputs; real-data tables were still produced.")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# tables: source make_publication_tables_v2.py
# ============================================================================
def tables(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['make_publication_tables_v2.py'] + list(argv or [])
    try:
        """Render publication V2 CSV tables as compact Markdown and LaTeX tables."""
        import argparse
        from pathlib import Path
        import numpy as np
        import pandas as pd


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--tables_dir", default="outputs/publication_v2/tables")
            return p.parse_args()


        def display_digits(col):
            c = str(col).lower()
            if c in {"chunk_idx", "n", "n_boundaries"}:
                return 0
            if "mse" in c:
                return 6
            if "r2" in c or "r²" in c or c.endswith("_r"):
                return 4
            if "improvement" in c or "delta" in c:
                return 4
            if "condition" in c or "valid_fraction" in c:
                return 3
            if "boundary" in c:
                return 3
            if "lambda" in c:
                return 2
            if any(k in c for k in ["d_e", "w_ee", "selected_h", "ci95", "estimate"]):
                return 4
            return 4


        def clean_cell(x, col=None):
            if pd.isna(x):
                return ""
            if isinstance(x, (float, np.floating)):
                d = display_digits(col)
                return f"{float(x):.{d}f}"
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            return str(x)


        def markdown_table(df):
            headers = [str(c) for c in df.columns]
            rows = [[clean_cell(v, c) for v, c in zip(row, df.columns)] for row in df.to_numpy()]
            widths = [len(h) for h in headers]
            for row in rows:
                widths = [max(w, len(v)) for w, v in zip(widths, row)]
            line1 = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
            line2 = "| " + " | ".join("-" * w for w in widths) + " |"
            body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
            return "\n".join([line1, line2] + body) + "\n"


        def latex_escape(s):
            s = str(s)
            repl = {
                "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                "±": r"$\pm$", "×": r"$\times$", "σ": r"$\sigma$",
                "λ": r"$\lambda$", "μ": r"$\mu$", "²": r"$^2$",
                "−": "-", "→": r"$\rightarrow$",
            }
            for a, b in repl.items():
                s = s.replace(a, b)
            return s


        def latex_table(df):
            cols = "l" * len(df.columns)
            lines = [r"\begin{tabular}{" + cols + "}", r"\hline"]
            lines.append(" & ".join(latex_escape(c) for c in df.columns) + r" \\")
            lines.append(r"\hline")
            for row in df.itertuples(index=False):
                vals = [clean_cell(v, c) for v, c in zip(row, df.columns)]
                lines.append(" & ".join(latex_escape(v) for v in vals) + r" \\")
            lines += [r"\hline", r"\end{tabular}"]
            return "\n".join(lines) + "\n"


        def main():
            args = parse_args()
            td = Path(args.tables_dir)
            if not td.exists():
                raise FileNotFoundError(td)
            csvs = sorted(td.glob("*.csv"))
            if not csvs:
                raise RuntimeError(f"No CSV tables found in {td}")
            for csv_path in csvs:
                df = pd.read_csv(csv_path)
                csv_path.with_suffix(".md").write_text(markdown_table(df))
                csv_path.with_suffix(".tex").write_text(latex_table(df))
                print(f"Rendered {csv_path.name} -> .md, .tex")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# figures: source make_publication_figures_v2.py
# ============================================================================
def figures(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['make_publication_figures_v2.py'] + list(argv or [])
    try:
        """
        Publication V2 figures for the regression-assisted, model-selective sequential
        Wilson-Cowan gPINN study.

        No training is performed.

        Main changes from V1
        --------------------
        1. Figure 1 uses a two-lane conceptual layout separating parameter inference
           from neural-state estimation.
        2. Figure 2C no longer connects unrelated physical parameters with lines.
        3. Figure 2D evaluates the single robust smoothing configuration across noise
           levels instead of showing an oracle "best at each noise" curve.
        4. Figure 3 focuses entirely on physical inference/model selection.
        5. Figure 4 uses a common observed/predicted color scale, restores absolute
           error maps, adds session-wide R², and moves the PDE residual to Supplement S1.
        6. Figure 5 highlights lambda_cont=0.05 and clarifies boundary labeling.
        """
        import argparse
        import json
        import sys
        from pathlib import Path

        import numpy as np
        import pandas as pd

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        import torch


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--project_root", default=".")
            p.add_argument("--master_dir", default="outputs/publication_v2/master")
            p.add_argument("--final_run_dir", default="outputs/final_dual_model_sequential20_cont005")
            p.add_argument("--data_dir", default="data/processed")
            p.add_argument("--subject_prefix", default="sub-GC48_ses-day0")
            p.add_argument("--representative_chunk", type=int, default=12)
            p.add_argument("--output_dir", default="outputs/publication_v2/figures")
            p.add_argument("--dpi", type=int, default=300)
            p.add_argument("--scatter_points", type=int, default=100000)
            p.add_argument("--seed", type=int, default=42)
            return p.parse_args()


        def save_both(fig, pathstem: Path, dpi: int):
            pathstem.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pathstem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(pathstem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)


        def panel_label(ax, label):
            ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom")


        def add_box(ax, xy, w, h, text, fontsize=9.5, lw=1.3):
            x, y = xy
            patch = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                fill=False, linewidth=lw,
            )
            ax.add_patch(patch)
            ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
            return patch


        def arrow(ax, a, b, **kwargs):
            ax.annotate("", xy=b, xytext=a,
                        arrowprops=dict(arrowstyle="->", lw=1.35, **kwargs))


        def figure1_pipeline(out_dir: Path, dpi: int):
            fig, ax = plt.subplots(figsize=(13.5, 7.6))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            ax.text(0.03, 0.91, "A  Parameter inference and model selection",
                    fontsize=12, fontweight="bold")
            ax.text(0.03, 0.48, "B  Neural-state estimation and sequential transfer",
                    fontsize=12, fontweight="bold")

            # Lane A
            add_box(ax, (0.04, 0.68), 0.17, 0.16,
                    "30-s calcium chunk\n64×64, 30 Hz\nsession-normalized")
            add_box(ax, (0.28, 0.68), 0.19, 0.16,
                    "Resolution-aware smoothing\n+ finite-difference\npseudo-derivatives")
            add_box(ax, (0.54, 0.68), 0.18, 0.16,
                    "Full vs reduced\nlogit regression\n+ uncertainty checks")
            add_box(ax, (0.79, 0.68), 0.17, 0.16,
                    "Evidence-based\nmodel selection\nfull or $D_E=0$")
            arrow(ax, (0.21, 0.76), (0.28, 0.76))
            arrow(ax, (0.47, 0.76), (0.54, 0.76))
            arrow(ax, (0.72, 0.76), (0.79, 0.76))

            # Lane B
            add_box(ax, (0.79, 0.27), 0.17, 0.15,
                    "Fix current-chunk\n$(w_{EE},D_E,h)$\nduring neural training")
            add_box(ax, (0.54, 0.27), 0.18, 0.15,
                    "Stage A: state fit\nStage B: physics +\npseudo-derivative training")
            add_box(ax, (0.29, 0.27), 0.18, 0.15,
                    "Sequential transfer\ncarry neural-state\nweights only")
            add_box(ax, (0.04, 0.27), 0.17, 0.15,
                    "Boundary regularization\n$\\lambda_{cont}=0.05$\nfrom chunk 1 onward")
            arrow(ax, (0.875, 0.68), (0.875, 0.42))
            arrow(ax, (0.79, 0.345), (0.72, 0.345))
            arrow(ax, (0.54, 0.345), (0.47, 0.345))
            arrow(ax, (0.29, 0.345), (0.21, 0.345))

            # Next chunk loop.
            ax.annotate(
                "next chunk",
                xy=(0.125, 0.68), xytext=(0.125, 0.17),
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.62", lw=1.3),
                fontsize=9.5,
            )

            ax.text(
                0.50, 0.08,
                "Physical coefficients are re-estimated independently in every chunk; "
                "optimizer state is reset. Only the neural-state representation is propagated.",
                ha="center", va="center", fontsize=10.5
            )
            ax.set_title(
                "Regression-assisted, model-selective sequential Wilson–Cowan gPINN",
                fontsize=15, pad=10
            )
            save_both(fig, out_dir / "figure1_final_pipeline_v2", dpi)


        def figure2_synthetic(master_dir: Path, out_dir: Path, dpi: int):
            runs_path = master_dir / "synthetic_recovery_runs.csv"
            diag_path = master_dir / "synthetic_derivative_diagnostic.json"
            methods_path = master_dir / "synthetic_method_comparison.csv"
            sens_path = master_dir / "synthetic_smoothing_sensitivity.json"
            smooth_path = master_dir / "synthetic_smoothing_summary.csv"

            if not runs_path.exists() or not diag_path.exists():
                print("WARNING: Figure 2 skipped: required synthetic files missing.")
                return

            runs = pd.read_csv(runs_path)
            diag = json.loads(diag_path.read_text())

            fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

            # A: baseline joint inverse failure
            ax = axes[0, 0]
            panel_label(ax, "A")
            noises = sorted(runs["noise_sigma"].unique())
            x = np.arange(len(noises))
            width = 0.24
            fields = [
                ("D_E_relerr_pct", r"$D_E$"),
                ("w_EE_relerr_pct", r"$w_{EE}$"),
                ("h_relerr_pct", r"$h$"),
            ]
            for j, (field, label) in enumerate(fields):
                means = [runs.loc[runs.noise_sigma == n, field].mean() for n in noises]
                ax.bar(x + (j - 1) * width, means, width=width, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{n:g}" for n in noises])
            ax.set_xlabel(r"Added noise $\sigma$")
            ax.set_ylabel("Mean parameter relative error (%)")
            ax.set_title("Baseline joint gPINN recovers state but not parameters")
            ax.legend(frameon=False)

            # B: derivative fidelity
            ax = axes[0, 1]
            panel_label(ax, "B")
            metrics = [
                ("state_E_metrics", r"$E$"),
                ("time_derivative_Et_metrics", r"$E_t$"),
                ("laplacian_metrics", r"$\nabla^2 E$"),
            ]
            r2 = [float(diag.get(k, {}).get("r2", np.nan)) for k, _ in metrics]
            ax.bar(np.arange(3), r2)
            ax.axhline(0, linewidth=1)
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels([lab for _, lab in metrics])
            ax.set_ylabel(r"$R^2$ against synthetic truth")
            ax.set_title("High state fidelity does not imply derivative fidelity")

            # C: scale-normalized method comparison (no connecting lines)
            ax = axes[1, 0]
            panel_label(ax, "C")
            if methods_path.exists():
                m = pd.read_csv(methods_path)
                truth_row = m[m["Method"] == "Truth"]
                if len(truth_row) == 1:
                    truth = truth_row.iloc[0]
                    m = m[m["Method"] != "Truth"].copy()
                    params = ["D_E", "w_EE", "h"]
                    labels = [r"$D_E$", r"$w_{EE}$", r"$h$"]
                    xx = np.arange(len(params), dtype=float)
                    offsets = np.linspace(-0.24, 0.24, max(len(m), 1))
                    for off, (_, row) in zip(offsets, m.iterrows()):
                        ratios = []
                        for p in params:
                            tv = float(truth[p])
                            ratios.append(float(row[p]) / tv if abs(tv) > 1e-12 else np.nan)
                        ax.scatter(xx + off, ratios, s=48, label=row["Method"])
                    ax.axhline(1.0, linestyle="--", linewidth=1, label="Truth")
                    ax.set_xticks(xx)
                    ax.set_xticklabels(labels)
                    ax.set_ylabel("Estimate / truth")
                    ax.set_title("Derivative-informed estimation improves recoverability")
                    ax.legend(frameon=False, fontsize=8)
                else:
                    ax.axis("off")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "Method-comparison data unavailable", ha="center", va="center")

            # D: fixed robust smoothing choice, not oracle best-by-noise
            ax = axes[1, 1]
            panel_label(ax, "D")
            if sens_path.exists() and smooth_path.exists():
                sens = json.loads(sens_path.read_text())
                sm = pd.read_csv(smooth_path)
                robust = sens.get("robust_best_nonzero_noise", {})
                st = float(robust.get("smooth_sigma_t_frames", np.nan))
                sx = float(robust.get("smooth_sigma_space_pixels", np.nan))
                sub = sm[
                    np.isclose(sm["smooth_sigma_t_frames"].astype(float), st) &
                    np.isclose(sm["smooth_sigma_space_pixels"].astype(float), sx)
                ].sort_values("noise_sigma")
                if not sub.empty:
                    ax.plot(sub["noise_sigma"], sub["mean_param_relerr_mean_pct"], marker="o")
                    ax.set_xlabel(r"Added noise $\sigma$")
                    ax.set_ylabel("Mean parameter relative error (%)")
                    ax.set_title(
                        rf"Fixed robust smoothing: $\sigma_t={st:g}$ frames, "
                        rf"$\sigma_{{xy}}={sx:g}$ px"
                    )
                else:
                    ax.axis("off")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "Fixed-smoothing sensitivity data unavailable",
                        ha="center", va="center")

            fig.suptitle(
                "Synthetic validation: state accuracy alone is insufficient for inverse parameter recovery",
                fontsize=14
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            save_both(fig, out_dir / "figure2_synthetic_recoverability_v2", dpi)


        def figure3_real_model(master_dir: Path, out_dir: Path, dpi: int):
            p = master_dir / "final_continuity_chunkwise.csv"
            if not p.exists():
                raise FileNotFoundError(p)
            d = pd.read_csv(p)
            chunks = d["chunk_idx"].to_numpy(int)
            mins = d["time_start_min"].to_numpy(float) if "time_start_min" in d else chunks * 0.5

            fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

            # A: D_E with uncertainty
            ax = axes[0, 0]
            panel_label(ax, "A")
            D = d["full_regression_D_E"].to_numpy(float)
            lo = d["D_boot_ci95_low"].to_numpy(float)
            hi = d["D_boot_ci95_high"].to_numpy(float)
            yerr = np.vstack([D - lo, hi - D])
            ax.errorbar(mins, D, yerr=yerr, fmt="o-", capsize=2, linewidth=1)
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("Session time (min)")
            ax.set_ylabel(r"Unrestricted $D_E$ estimate")
            ax.set_title("All 20 bootstrap intervals include zero")

            # B: w_EE
            ax = axes[0, 1]
            panel_label(ax, "B")
            w = d["selected_w_EE"].astype(float)
            ax.plot(mins, w, marker="o")
            ax.axhline(w.mean(), linestyle="--", linewidth=1,
                       label=rf"mean={w.mean():.3f}")
            ax.set_xlabel("Session time (min)")
            ax.set_ylabel(r"$w_{EE}$")
            ax.set_title("Reduced-model recurrent excitation")
            ax.legend(frameon=False)

            # C: h
            ax = axes[1, 0]
            panel_label(ax, "C")
            h = d["selected_h"].astype(float)
            ax.plot(mins, h, marker="s")
            ax.axhline(h.mean(), linestyle="--", linewidth=1,
                       label=rf"mean={h.mean():.3f}")
            ax.set_xlabel("Session time (min)")
            ax.set_ylabel(r"$h$")
            ax.set_title("Reduced-model external drive")
            ax.legend(frameon=False)

            # D: full-vs-reduced evidence
            ax = axes[1, 1]
            panel_label(ax, "D")
            ax.plot(mins, d["nested_logit_improvement_pct"], marker="o", label="Logit")
            ax.plot(mins, d["nested_pde_improvement_pct"], marker="s", label="PDE")
            ax.axhline(0.1, linestyle="--", linewidth=1, label="0.1% selection threshold")
            ax.axhline(0, linewidth=0.8)
            ax.set_xlabel("Session time (min)")
            ax.set_ylabel("Full-model held-out improvement (%)")
            ax.set_title(r"Adding $D_E\nabla^2 E$ provides negligible held-out gain")
            ax.legend(frameon=False)

            fig.suptitle(
                r"Resting-state session: reduced $D_E=0$ model selected in all 20 chunks",
                fontsize=14
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            save_both(fig, out_dir / "figure3_real_model_selection_v2", dpi)


        def load_final_model(project_root: Path, final_run_dir: Path, chunk_idx: int, device):
            sys.path.insert(0, str(project_root))
            from src.training import FixedPhysicsGPINN

            ckpts = sorted((final_run_dir / "checkpoints").glob(f"seq_chunk_{chunk_idx:02d}_*.pt"))
            if len(ckpts) != 1:
                raise FileNotFoundError(f"Expected one final checkpoint for chunk {chunk_idx}; found {ckpts}")
            ck = torch.load(ckpts[0], map_location=device)
            fp = ck["fixed_parameters"]
            model = FixedPhysicsGPINN(
                model_kind=ck["model_kind"],
                w_EE=float(fp["w_EE"]), D_E=float(fp["D_E"]), h=float(fp["h"]),
                ff_dim=128, hidden_neurons=256, hidden_layers=6,
                scale_xy=25.0, scale_t=50.0,
            ).to(device)
            model.load_state_dict(ck["model_state"])
            model.eval()
            return model, ck, ckpts[0]


        def frame_coords(ti: int, T: int, H: int, W: int):
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            x = -1.0 + 2.0 * xx.ravel() / float(W - 1)
            y = -1.0 + 2.0 * yy.ravel() / float(H - 1)
            t = np.full(H * W, -1.0 + 2.0 * ti / float(T - 1))
            return np.stack([x, y, t], axis=1).astype(np.float32)


        def predict_points(model, X, device, batch=16384):
            out = []
            with torch.no_grad():
                for s in range(0, len(X), batch):
                    xb = torch.tensor(X[s:s + batch], dtype=torch.float32, device=device)
                    out.append(model.forward_E(xb).detach().cpu().numpy().ravel())
            return np.concatenate(out)


        def compute_pde_residual_map(project_root, model, X, H, W, device):
            sys.path.insert(0, str(project_root))
            from src.training import gpinn_residual_with_grad
            parts = []
            for s in range(0, len(X), 2048):
                xb = torch.tensor(X[s:s + 2048], dtype=torch.float32, device=device)
                f, _, _ = gpinn_residual_with_grad(
                    model, xb, tau=1.0, chunk_duration=30.0, l_fov=10.0
                )
                parts.append(f.detach().cpu().numpy().ravel())
            return np.concatenate(parts).reshape(H, W)


        def figure4_reconstruction(project_root: Path, master_dir: Path,
                                   final_run_dir: Path, data_dir: Path,
                                   subject_prefix: str, chunk_idx: int,
                                   out_dir: Path, dpi: int,
                                   scatter_points: int, seed: int):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, ck, ckpt_path = load_final_model(
                project_root, final_run_dir, chunk_idx, device
            )
            data_path = data_dir / f"{subject_prefix}_chunk-{chunk_idx:02d}_dfof.npz"
            raw = np.load(data_path)["dfof"].astype(np.float64)
            gmin = float(ck["normalization"]["global_min"])
            gmax = float(ck["normalization"]["global_max"])
            E = (raw - gmin) / (gmax - gmin + 1e-8)
            T, H, W = E.shape

            frames = [0, T // 2, T - 1]
            pred_frames = {}
            for ti in frames:
                pred_frames[ti] = predict_points(
                    model, frame_coords(ti, T, H, W), device
                ).reshape(H, W)

            # One common scale for every observed/predicted state map.
            state_min = min(
                min(float(E[ti].min()) for ti in frames),
                min(float(pred_frames[ti].min()) for ti in frames),
            )
            state_max = max(
                max(float(E[ti].max()) for ti in frames),
                max(float(pred_frames[ti].max()) for ti in frames),
            )
            errors = {ti: np.abs(E[ti] - pred_frames[ti]) for ti in frames}
            error_max = max(float(errors[ti].max()) for ti in frames)

            # Spatial mean time course.
            pred_mean = np.empty(T, dtype=float)
            for ti in range(T):
                pred_mean[ti] = predict_points(
                    model, frame_coords(ti, T, H, W), device
                ).mean()
            obs_mean = E.mean(axis=(1, 2))
            seconds = np.arange(T) / 30.0

            # Dense pointwise comparison.
            rng = np.random.default_rng(seed)
            total = T * H * W
            n = min(scatter_points, total)
            flat = rng.choice(total, size=n, replace=False)
            ti = flat // (H * W)
            rem = flat % (H * W)
            yi = rem // W
            xi = rem % W
            Xs = np.column_stack([
                -1 + 2 * xi / (W - 1),
                -1 + 2 * yi / (H - 1),
                -1 + 2 * ti / (T - 1),
            ]).astype(np.float32)
            ys = E[ti, yi, xi]
            ps = predict_points(model, Xs, device)

            # Session-wide R² from the locked final results.
            final_master = pd.read_csv(master_dir / "final_continuity_chunkwise.csv")
            r2_chunks = final_master["random_val_state_r2"].astype(float)
            mins = (final_master["time_start_min"].astype(float)
                    if "time_start_min" in final_master
                    else final_master["chunk_idx"].astype(float) * 0.5)

            fig, axes = plt.subplots(4, 3, figsize=(12.2, 14.0))

            # A: observed
            for j, ti0 in enumerate(frames):
                ax = axes[0, j]
                im_state = ax.imshow(
                    E[ti0], origin="lower", vmin=state_min, vmax=state_max
                )
                ax.set_title(f"Observed, t={ti0/30:.1f} s")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    panel_label(ax, "A")
            cb = fig.colorbar(im_state, ax=axes[0, :].tolist(),
                              fraction=0.018, pad=0.02)
            cb.set_label("Normalized activity")

            # B: predicted, exact same scale
            for j, ti0 in enumerate(frames):
                ax = axes[1, j]
                im_state2 = ax.imshow(
                    pred_frames[ti0], origin="lower", vmin=state_min, vmax=state_max
                )
                ax.set_title(f"Predicted, t={ti0/30:.1f} s")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    panel_label(ax, "B")
            cb = fig.colorbar(im_state2, ax=axes[1, :].tolist(),
                              fraction=0.018, pad=0.02)
            cb.set_label("Normalized activity")

            # C: absolute error
            for j, ti0 in enumerate(frames):
                ax = axes[2, j]
                im_err = ax.imshow(
                    errors[ti0], origin="lower", vmin=0, vmax=error_max
                )
                ax.set_title(f"|Observed − predicted|, t={ti0/30:.1f} s")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    panel_label(ax, "C")
            cb = fig.colorbar(im_err, ax=axes[2, :].tolist(),
                              fraction=0.018, pad=0.02)
            cb.set_label("Absolute error")

            # D: mean time course
            ax = axes[3, 0]
            panel_label(ax, "D")
            ax.plot(seconds, obs_mean, label="Observed")
            ax.plot(seconds, pred_mean, label="Predicted")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Spatial mean normalized activity")
            ax.set_title("Spatially averaged activity")
            ax.legend(frameon=False)

            # E: pointwise density scatter
            ax = axes[3, 1]
            panel_label(ax, "E")
            hb = ax.hexbin(ys, ps, gridsize=60, mincnt=1)
            lim = [min(ys.min(), ps.min()), max(ys.max(), ps.max())]
            ax.plot(lim, lim, linestyle="--", linewidth=1)
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.set_title("Pointwise reconstruction")
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Count per hexagon")

            # F: session-wide interpolation fidelity
            ax = axes[3, 2]
            panel_label(ax, "F")
            ax.plot(mins, r2_chunks, marker="o")
            ax.axhline(r2_chunks.mean(), linestyle="--", linewidth=1,
                       label=rf"mean={r2_chunks.mean():.3f}")
            ax.set_xlabel("Session time (min)")
            ax.set_ylabel(r"Random-pointwise validation $R^2$")
            ax.set_title("Session-wide interpolation fidelity")
            ax.legend(frameon=False)

            fig.suptitle(
                f"Final reduced-model gPINN reconstruction: representative chunk {chunk_idx}",
                fontsize=14
            )
            fig.subplots_adjust(top=0.95, hspace=0.38, wspace=0.28)
            save_both(fig, out_dir / "figure4_reconstruction_diagnostics_v2", dpi)

            # Supplement S1: signed PDE residual map.
            ti_mid = T // 2
            residual = compute_pde_residual_map(
                project_root, model, frame_coords(ti_mid, T, H, W), H, W, device
            )
            vmax = float(np.max(np.abs(residual)))
            fig_s, ax_s = plt.subplots(figsize=(5.6, 5.0))
            im = ax_s.imshow(residual, origin="lower", vmin=-vmax, vmax=vmax)
            ax_s.set_xticks([]); ax_s.set_yticks([])
            ax_s.set_title(f"Wilson–Cowan PDE residual, chunk {chunk_idx}, t={ti_mid/30:.1f} s")
            cb = fig_s.colorbar(im, ax=ax_s, fraction=0.046, pad=0.04)
            cb.set_label("PDE residual")
            save_both(fig_s, out_dir / "supp_figure_s1_pde_residual_v2", dpi)

            print(f"Figure 4 used checkpoint: {ckpt_path}")


        def figure5_sequential(master_dir: Path, out_dir: Path, dpi: int):
            final_path = master_dir / "final_continuity_chunkwise.csv"
            noc_path = master_dir / "no_continuity_chunkwise.csv"
            ab_path = master_dir / "continuity_ablation_summary.csv"
            if not final_path.exists() or not noc_path.exists():
                print("WARNING: Figure 5 skipped because sequential master files are missing.")
                return

            final = pd.read_csv(final_path)
            noc = pd.read_csv(noc_path)
            ab = pd.read_csv(ab_path) if ab_path.exists() else pd.DataFrame()

            fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))

            ax = axes[0, 0]
            panel_label(ax, "A")
            d = noc[noc["chunk_idx"] > 0]
            ax.plot(d["chunk_idx"], d["pretrain_random_val_state_r2"], marker="o")
            ax.set_xlabel("Chunk")
            ax.set_ylabel(r"Pre-training validation $R^2$")
            ax.set_title("Inherited neural-state representation before retraining")

            ax = axes[0, 1]
            panel_label(ax, "B")
            if "sequential_minus_independent_r2" in noc.columns:
                m = noc[
                    noc["sequential_minus_independent_r2"].notna() &
                    (noc["chunk_idx"] > 0)
                ]
                ax.bar(m["chunk_idx"].astype(str), m["sequential_minus_independent_r2"])
                ax.axhline(0, linewidth=1)
                ax.set_xlabel("Matched chunk")
                ax.set_ylabel(r"$\Delta R^2$: sequential − independent")
                ax.set_title("Neural-state warm-start effect")
            else:
                ax.axis("off")

            ax = axes[1, 0]
            panel_label(ax, "C")
            if not ab.empty:
                ax.plot(ab["lambda_cont"], ab["mean_boundary_ratio"],
                        marker="o", label="Mean boundary ratio")
                ax.axhline(1, linestyle="--", linewidth=1,
                           label="Observed transition scale")
                ax.axvline(0.05, linestyle=":", linewidth=1,
                           label=r"Selected $\lambda_{cont}=0.05$")
                sel = ab[np.isclose(ab["lambda_cont"].astype(float), 0.05)]
                if len(sel) == 1 and "mean_delta_r2_vs_lambda0" in sel.columns:
                    dr2 = float(sel.iloc[0]["mean_delta_r2_vs_lambda0"])
                    ax.text(
                        0.05, float(sel.iloc[0]["mean_boundary_ratio"]),
                        rf"  $\Delta R^2={dr2:+.5f}$",
                        va="bottom", fontsize=8.5
                    )
                ax.set_xlabel(r"Continuity weight $\lambda_{cont}$")
                ax.set_ylabel("Mean predicted / observed boundary RMSE")
                ax.set_title("Continuity-weight ablation")
                ax.legend(frameon=False, fontsize=8.5)
            else:
                ax.axis("off")

            ax = axes[1, 1]
            panel_label(ax, "D")
            merged = noc[["chunk_idx", "boundary_pred_to_observed_ratio"]].merge(
                final[["chunk_idx", "boundary_pred_to_observed_ratio"]],
                on="chunk_idx", suffixes=("_no", "_cont")
            )
            merged = merged[merged["chunk_idx"] > 0]
            ax.plot(merged["chunk_idx"],
                    merged["boundary_pred_to_observed_ratio_no"],
                    marker="o", label="No continuity")
            ax.plot(merged["chunk_idx"],
                    merged["boundary_pred_to_observed_ratio_cont"],
                    marker="s", label=r"$\lambda_{cont}=0.05$")
            ax.axhline(1, linestyle="--", linewidth=1,
                       label="Observed transition scale")
            ax.set_xlabel(r"Boundary at start of chunk $i$")
            ax.set_ylabel("Predicted / observed boundary RMSE")
            ax.set_title("Boundary consistency across all 19 transitions")
            ax.legend(frameon=False, fontsize=8.5)

            fig.suptitle(
                "Sequential neural-state transfer and continuity regularization",
                fontsize=14
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            save_both(fig, out_dir / "figure5_sequential_continuity_v2", dpi)


        def main():
            args = parse_args()
            root = Path(args.project_root).resolve()
            master = root / args.master_dir
            final_run = root / args.final_run_dir
            data_dir = root / args.data_dir
            out_dir = root / args.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            figure1_pipeline(out_dir, args.dpi)
            figure2_synthetic(master, out_dir, args.dpi)
            figure3_real_model(master, out_dir, args.dpi)
            figure4_reconstruction(
                root, master, final_run, data_dir,
                args.subject_prefix, args.representative_chunk,
                out_dir, args.dpi, args.scatter_points, args.seed,
            )
            figure5_sequential(master, out_dir, args.dpi)
            print(f"Publication V2 figures saved to {out_dir}")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# final_figures: source make_publication_figures_final_hexbin.py
# ============================================================================
def final_figures(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['make_publication_figures_final_hexbin.py'] + list(argv or [])
    try:
        """
        Final publication figures for the regression-assisted, model-selective
        sequential Wilson-Cowan gPINN study.

        This script regenerates ONLY the two figures that still required layout work:
        Figure 1 (pipeline schematic) and Figure 4 (reconstruction diagnostics), plus
        Supplementary Figure S1 (PDE residual). It performs no training and does not
        modify any locked numerical results.
        """
        import argparse
        import sys
        from pathlib import Path

        import numpy as np
        import pandas as pd

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        import torch


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--project_root", default=".")
            p.add_argument("--master_dir", default="outputs/publication_final/master")
            p.add_argument("--final_run_dir",
                           default="outputs/final_dual_model_sequential20_cont005")
            p.add_argument("--data_dir", default="data/processed")
            p.add_argument("--subject_prefix", default="sub-GC48_ses-day0")
            p.add_argument("--representative_chunk", type=int, default=12)
            p.add_argument("--output_dir", default="outputs/publication_final/figures")
            p.add_argument("--dpi", type=int, default=300)
            p.add_argument("--scatter_points", type=int, default=100000)
            p.add_argument("--seed", type=int, default=42)
            return p.parse_args()


        def save_both(fig, pathstem: Path, dpi: int):
            pathstem.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pathstem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(pathstem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)


        def panel_label(ax, label, x=-0.10, y=1.04):
            ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom", ha="left")


        def add_box(ax, x, y, w, h, text, fontsize=10):
            box = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                fill=False, linewidth=1.35,
            )
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                    fontsize=fontsize, linespacing=1.15)
            return box


        def arrow(ax, x0, y0, x1, y1, connectionstyle=None):
            props = dict(arrowstyle="->", lw=1.35, shrinkA=2, shrinkB=2)
            if connectionstyle is not None:
                props["connectionstyle"] = connectionstyle
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=props)


        def figure1_pipeline(out_dir: Path, dpi: int):
            """
            Compact two-lane schematic:
              A. independent per-chunk parameter inference/model selection
              B. neural-state training + sequential transfer
            """
            fig, ax = plt.subplots(figsize=(13.2, 6.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            fig.suptitle(
                "Regression-assisted, model-selective sequential Wilson–Cowan gPINN",
                fontsize=16, y=0.97
            )

            # Lane headings.
            ax.text(0.035, 0.845, "A", fontsize=12, fontweight="bold", va="center")
            ax.text(0.060, 0.845, "Per-chunk parameter inference and model selection",
                    fontsize=12, fontweight="bold", va="center")

            ax.text(0.035, 0.415, "B", fontsize=12, fontweight="bold", va="center")
            ax.text(0.060, 0.415, "Neural-state estimation and sequential transfer",
                    fontsize=12, fontweight="bold", va="center")

            # Top lane: independent physical inference.
            xs = [0.055, 0.285, 0.515, 0.745]
            w = 0.19
            h = 0.17
            y_top = 0.61

            add_box(ax, xs[0], y_top, w, h,
                    "30-s calcium chunk\n$E_i(x,y,t)$\n64×64, 30 Hz")
            add_box(ax, xs[1], y_top, w, h,
                    "Resolution-aware smoothing\n+ finite-difference\npseudo-derivatives")
            add_box(ax, xs[2], y_top, w, h,
                    "Full vs reduced\nlogit regression\n+ uncertainty / conditioning")
            add_box(ax, xs[3], y_top, w, h,
                    "Evidence-based selection\nand current-chunk\n$(\\hat w_{EE},\\hat D_E,\\hat h)$")

            for j in range(3):
                arrow(ax, xs[j] + w, y_top + h/2,
                      xs[j+1], y_top + h/2)

            # Bottom lane: state representation.
            y_bot = 0.17
            xb = [0.055, 0.305, 0.555, 0.795]
            wb = [0.19, 0.19, 0.17, 0.15]
            hb = 0.16

            add_box(ax, xb[0], y_bot, wb[0], hb,
                    "Stage A\nstate-only fit\n(current chunk)")
            add_box(ax, xb[1], y_bot, wb[1], hb,
                    "Stage B\nfixed-physics +\npseudo-derivative refinement")
            add_box(ax, xb[2], y_bot, wb[2], hb,
                    "Final neural-state\nrepresentation\n$E_{\\theta_i}$")
            add_box(ax, xb[3], y_bot, wb[3], hb,
                    "Next chunk\ninitialize with\n$\\theta_i$ only")

            arrow(ax, xb[0] + wb[0], y_bot + hb/2,
                  xb[1], y_bot + hb/2)
            arrow(ax, xb[1] + wb[1], y_bot + hb/2,
                  xb[2], y_bot + hb/2)
            arrow(ax, xb[2] + wb[2], y_bot + hb/2,
                  xb[3], y_bot + hb/2)

            # Data directly supplies Stage A.
            arrow(ax, xs[0] + w/2, y_top,
                  xb[0] + wb[0]/2, y_bot + hb)

            # Selected physical coefficients supply Stage B.
            # Route on the far-right side, below the B-lane heading, so the connector
            # never crosses text.
            sel_x = xs[3] + w/2
            stageb_right = xb[1] + wb[1]
            route_y = 0.365

            ax.annotate(
                "", xy=(sel_x, route_y), xytext=(sel_x, y_top),
                arrowprops=dict(arrowstyle="-", lw=1.35)
            )
            ax.annotate(
                "", xy=(stageb_right, route_y), xytext=(sel_x, route_y),
                arrowprops=dict(arrowstyle="-", lw=1.35)
            )
            arrow(ax, stageb_right, route_y,
                  stageb_right, y_bot + hb)

            # Continuity constraint belongs to the transition between adjacent chunks.
            ax.text(
                0.715, 0.105,
                r"chunk boundary:  $\mathcal{L}_{cont}$,  $\lambda_{cont}=0.05$",
                fontsize=9.5, ha="center", va="center"
            )

            # Explicitly state what does and does not propagate; this is central to the method.
            ax.text(
                0.50, 0.035,
                "Physical coefficients are re-estimated independently for each chunk; "
                "optimizer state is reset. Only neural weights are warm-started.",
                fontsize=9.5, ha="center", va="center"
            )

            save_both(fig, out_dir / "figure1_final_pipeline", dpi)


        def load_final_model(project_root: Path, final_run_dir: Path,
                             chunk_idx: int, device):
            sys.path.insert(0, str(project_root))
            from src.training import FixedPhysicsGPINN

            ckpts = sorted(
                (final_run_dir / "checkpoints").glob(
                    f"seq_chunk_{chunk_idx:02d}_*.pt"
                )
            )
            if len(ckpts) != 1:
                raise FileNotFoundError(
                    f"Expected one checkpoint for chunk {chunk_idx}; found {ckpts}"
                )

            ck = torch.load(ckpts[0], map_location=device)
            fp = ck["fixed_parameters"]
            model = FixedPhysicsGPINN(
                model_kind=ck["model_kind"],
                w_EE=float(fp["w_EE"]),
                D_E=float(fp["D_E"]),
                h=float(fp["h"]),
                ff_dim=128,
                hidden_neurons=256,
                hidden_layers=6,
                scale_xy=25.0,
                scale_t=50.0,
            ).to(device)
            model.load_state_dict(ck["model_state"])
            model.eval()
            return model, ck, ckpts[0]


        def frame_coords(ti: int, T: int, H: int, W: int):
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            x = -1.0 + 2.0 * xx.ravel() / float(W - 1)
            y = -1.0 + 2.0 * yy.ravel() / float(H - 1)
            t = np.full(H * W, -1.0 + 2.0 * ti / float(T - 1))
            return np.stack([x, y, t], axis=1).astype(np.float32)


        def predict_points(model, X, device, batch=16384):
            out = []
            with torch.no_grad():
                for s in range(0, len(X), batch):
                    xb = torch.tensor(
                        X[s:s+batch], dtype=torch.float32, device=device
                    )
                    out.append(
                        model.forward_E(xb).detach().cpu().numpy().ravel()
                    )
            return np.concatenate(out)


        def compute_pde_residual_map(project_root: Path, model, X,
                                     H: int, W: int, device):
            sys.path.insert(0, str(project_root))
            from src.training import gpinn_residual_with_grad

            parts = []
            for s in range(0, len(X), 2048):
                xb = torch.tensor(
                    X[s:s+2048], dtype=torch.float32, device=device
                )
                f, _, _ = gpinn_residual_with_grad(
                    model, xb,
                    tau=1.0,
                    chunk_duration=30.0,
                    l_fov=10.0,
                )
                parts.append(f.detach().cpu().numpy().ravel())
            return np.concatenate(parts).reshape(H, W)


        def figure4_reconstruction(project_root: Path,
                                   master_dir: Path,
                                   final_run_dir: Path,
                                   data_dir: Path,
                                   subject_prefix: str,
                                   chunk_idx: int,
                                   out_dir: Path,
                                   dpi: int,
                                   scatter_points: int,
                                   seed: int):
            """
            Publication layout with dedicated colorbar columns and no panel overlap.

            Observed and predicted state maps share EXACTLY the same normalization.
            Absolute-error maps use a separate shared scale.
            """
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model, ck, ckpt_path = load_final_model(
                project_root, final_run_dir, chunk_idx, device
            )

            data_path = (
                data_dir /
                f"{subject_prefix}_chunk-{chunk_idx:02d}_dfof.npz"
            )
            raw = np.load(data_path)["dfof"].astype(np.float64)

            gmin = float(ck["normalization"]["global_min"])
            gmax = float(ck["normalization"]["global_max"])
            E = (raw - gmin) / (gmax - gmin + 1e-8)
            T, H, W = E.shape

            frames = [0, T // 2, T - 1]
            pred_frames = {
                ti: predict_points(
                    model, frame_coords(ti, T, H, W), device
                ).reshape(H, W)
                for ti in frames
            }

            # Shared state scale across ALL six observed/predicted maps.
            state_min = min(
                min(float(E[ti].min()) for ti in frames),
                min(float(pred_frames[ti].min()) for ti in frames),
            )
            state_max = max(
                max(float(E[ti].max()) for ti in frames),
                max(float(pred_frames[ti].max()) for ti in frames),
            )

            abs_errors = {
                ti: np.abs(E[ti] - pred_frames[ti])
                for ti in frames
            }
            error_max = max(
                float(abs_errors[ti].max()) for ti in frames
            )

            # Spatial mean trace.
            pred_mean = np.empty(T, dtype=float)
            for ti in range(T):
                pred_mean[ti] = predict_points(
                    model, frame_coords(ti, T, H, W), device
                ).mean()
            obs_mean = E.mean(axis=(1, 2))
            seconds = np.arange(T) / 30.0

            # Pointwise sample for a clean scatter diagnostic.
            rng = np.random.default_rng(seed)
            total = T * H * W
            n = min(scatter_points, total)
            flat = rng.choice(total, size=n, replace=False)
            ti_s = flat // (H * W)
            rem = flat % (H * W)
            yi_s = rem // W
            xi_s = rem % W
            Xs = np.column_stack([
                -1 + 2 * xi_s / (W - 1),
                -1 + 2 * yi_s / (H - 1),
                -1 + 2 * ti_s / (T - 1),
            ]).astype(np.float32)
            y_obs = E[ti_s, yi_s, xi_s]
            y_pred = predict_points(model, Xs, device)

            # Locked session-wide interpolation fidelity.
            session = pd.read_csv(
                master_dir / "final_continuity_chunkwise.csv"
            )
            r2 = session["random_val_state_r2"].astype(float).to_numpy()
            if "time_start_min" in session.columns:
                mins = session["time_start_min"].astype(float).to_numpy()
            else:
                mins = (
                    session["chunk_idx"].astype(float).to_numpy() * 0.5
                )

            fig = plt.figure(figsize=(12.8, 12.4), constrained_layout=True)
            gs = fig.add_gridspec(
                4, 4,
                width_ratios=[1, 1, 1, 0.055],
                height_ratios=[1, 1, 1, 0.95],
            )

            obs_axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
            pred_axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
            err_axes = [fig.add_subplot(gs[2, j]) for j in range(3)]

            cax_state = fig.add_subplot(gs[0:2, 3])
            cax_error = fig.add_subplot(gs[2, 3])

            ax_trace = fig.add_subplot(gs[3, 0])
            ax_scatter = fig.add_subplot(gs[3, 1])
            ax_r2 = fig.add_subplot(gs[3, 2])
            ax_blank = fig.add_subplot(gs[3, 3])
            ax_blank.axis("off")

            # A. Observed snapshots.
            im_state = None
            for j, ti0 in enumerate(frames):
                ax = obs_axes[j]
                im_state = ax.imshow(
                    E[ti0],
                    origin="lower",
                    vmin=state_min,
                    vmax=state_max,
                    interpolation="nearest",
                )
                ax.set_title(f"t = {ti0/30:.1f} s", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            panel_label(obs_axes[0], "A")
            obs_axes[0].set_ylabel("Observed", fontsize=11, labelpad=10)

            # B. Predicted snapshots, SAME state scale.
            for j, ti0 in enumerate(frames):
                ax = pred_axes[j]
                ax.imshow(
                    pred_frames[ti0],
                    origin="lower",
                    vmin=state_min,
                    vmax=state_max,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            panel_label(pred_axes[0], "B")
            pred_axes[0].set_ylabel("Predicted", fontsize=11, labelpad=10)

            cb = fig.colorbar(im_state, cax=cax_state)
            cb.set_label("Normalized activity")

            # C. Absolute error.
            im_err = None
            for j, ti0 in enumerate(frames):
                ax = err_axes[j]
                im_err = ax.imshow(
                    abs_errors[ti0],
                    origin="lower",
                    vmin=0,
                    vmax=error_max,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            panel_label(err_axes[0], "C")
            err_axes[0].set_ylabel("Absolute error", fontsize=11, labelpad=10)

            cb2 = fig.colorbar(im_err, cax=cax_error)
            cb2.set_label(r"$|E_{\mathrm{obs}}-E_{\mathrm{pred}}|$")

            # D. Mean time course.
            panel_label(ax_trace, "D")
            ax_trace.plot(seconds, obs_mean, label="Observed")
            ax_trace.plot(seconds, pred_mean, label="Predicted")
            ax_trace.set_xlabel("Time (s)")
            ax_trace.set_ylabel("Spatial mean activity")
            ax_trace.set_title("Spatially averaged activity", fontsize=10)
            ax_trace.legend(frameon=False, fontsize=8)

            # E. Pointwise reconstruction.
            # Dense space-time predictions are shown with a hexbin density plot to avoid
            # overplotting and make the concentration around the identity line visible.
            panel_label(ax_scatter, "E")
            hb = ax_scatter.hexbin(
                y_obs, y_pred,
                gridsize=55,
                mincnt=1,
            )
            lo = float(min(y_obs.min(), y_pred.min()))
            hi = float(max(y_obs.max(), y_pred.max()))
            ax_scatter.plot([lo, hi], [lo, hi],
                            linestyle="--", linewidth=1)
            ax_scatter.set_xlim(lo, hi)
            ax_scatter.set_ylim(lo, hi)
            ax_scatter.set_xlabel("Observed")
            ax_scatter.set_ylabel("Predicted")
            ax_scatter.set_title("Pointwise reconstruction", fontsize=10)
            cb_hb = fig.colorbar(
                hb, ax=ax_scatter,
                fraction=0.046, pad=0.03
            )
            cb_hb.set_label("Count per hexagon", fontsize=8)
            cb_hb.ax.tick_params(labelsize=7)

            # F. Session-wide interpolation fidelity.
            panel_label(ax_r2, "F")
            ax_r2.plot(mins, r2, marker="o")
            ax_r2.axhline(
                float(np.mean(r2)),
                linestyle="--",
                linewidth=1,
                label=rf"mean = {np.mean(r2):.3f}",
            )
            ax_r2.set_xlabel("Session time (min)")
            ax_r2.set_ylabel(r"Validation $R^2$")
            ax_r2.set_title("Session-wide interpolation fidelity", fontsize=10)
            ax_r2.legend(frameon=False, fontsize=8)

            fig.suptitle(
                f"Final reduced-model gPINN reconstruction: representative chunk {chunk_idx}",
                fontsize=14
            )

            save_both(
                fig,
                out_dir / "figure4_reconstruction_diagnostics",
                dpi
            )

            # Supplementary Figure S1: representative signed PDE residual.
            ti_mid = T // 2
            residual = compute_pde_residual_map(
                project_root,
                model,
                frame_coords(ti_mid, T, H, W),
                H, W, device,
            )
            vmax = float(np.max(np.abs(residual)))

            fig_s, ax_s = plt.subplots(figsize=(5.7, 5.2))
            im = ax_s.imshow(
                residual,
                origin="lower",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            ax_s.set_title(
                f"Representative PDE residual: chunk {chunk_idx}, t={ti_mid/30:.1f} s",
                fontsize=11
            )
            cb_s = fig_s.colorbar(im, ax=ax_s, fraction=0.046, pad=0.04)
            cb_s.set_label("Signed PDE residual")
            save_both(
                fig_s,
                out_dir / "supp_figure_s1_pde_residual",
                dpi
            )

            print(f"Figure 4 used checkpoint: {ckpt_path}")


        def main():
            args = parse_args()

            root = Path(args.project_root).resolve()
            master = root / args.master_dir
            final_run = root / args.final_run_dir
            data_dir = root / args.data_dir
            out_dir = root / args.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            figure1_pipeline(out_dir, args.dpi)
            figure4_reconstruction(
                root,
                master,
                final_run,
                data_dir,
                args.subject_prefix,
                args.representative_chunk,
                out_dir,
                args.dpi,
                args.scatter_points,
                args.seed,
            )

            print(f"Final Figure 1, Figure 4, and Supplement S1 saved to {out_dir}")
        return main()
    finally:
        _sys.argv = _old_argv
