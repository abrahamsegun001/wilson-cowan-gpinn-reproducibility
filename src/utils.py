"""Small repository utilities: configuration, environment, and input validation."""
from __future__ import annotations
from pathlib import Path
import importlib
import json
import numpy as np
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_environment():
    packages = ["numpy", "scipy", "pandas", "matplotlib", "torch", "yaml"]
    out = {}
    for name in packages:
        mod = importlib.import_module(name)
        out[name] = getattr(mod, "__version__", "available")
    return out


def validate_processed_data(data_dir, prefix="sub-GC48_ses-day0", chunks=20, expected_shape=(900,64,64)):
    data_dir = Path(data_dir)
    rows=[]
    for i in range(int(chunks)):
        p=data_dir/f"{prefix}_chunk-{i:02d}_dfof.npz"
        if not p.exists():
            rows.append({"chunk":i,"path":str(p),"ok":False,"reason":"missing"})
            continue
        with np.load(p, allow_pickle=False) as z:
            if "dfof" not in z.files:
                rows.append({"chunk":i,"path":str(p),"ok":False,"reason":"missing key dfof"})
                continue
            x=z["dfof"]
            ok=(tuple(x.shape)==tuple(expected_shape) and np.isfinite(x).all())
            rows.append({"chunk":i,"path":str(p),"ok":bool(ok),"shape":list(x.shape),"finite_fraction":float(np.isfinite(x).mean())})
    return rows
