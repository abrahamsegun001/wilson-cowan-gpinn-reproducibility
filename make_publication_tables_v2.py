#!/usr/bin/env python3
"""Render publication V2 CSV tables as compact Markdown and LaTeX tables."""
from __future__ import annotations
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


if __name__ == "__main__":
    main()
