#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--data_dir', default='data/processed')
p.add_argument('--prefix', default='sub-GC48_ses-day0')
p.add_argument('--chunks', type=int, default=20)
a=p.parse_args()
root=Path(a.data_dir)
errors=[]
mins=[]; maxs=[]
for i in range(a.chunks):
    f=root/f'{a.prefix}_chunk-{i:02d}_dfof.npz'
    if not f.exists():
        errors.append(f'MISSING {f}')
        continue
    d=np.load(f)
    if 'dfof' not in d:
        errors.append(f'{f}: missing key dfof')
        continue
    x=d['dfof']
    if x.ndim != 3:
        errors.append(f'{f}: expected 3D [T,H,W], got {x.shape}')
    if x.shape != (900,64,64):
        print(f'WARNING {f.name}: manuscript working shape is (900,64,64), got {x.shape}')
    if not np.isfinite(x).all():
        errors.append(f'{f}: contains non-finite values')
    mins.append(float(np.nanmin(x))); maxs.append(float(np.nanmax(x)))
    print(f'OK {f.name}: shape={x.shape} min={mins[-1]:.6g} max={maxs[-1]:.6g}')
if errors:
    print('\nValidation errors:')
    for e in errors: print(' -', e)
    raise SystemExit(1)
print(f'\nValidated {len(mins)} chunks. Session raw processed range: [{min(mins):.6g}, {max(maxs):.6g}]')
