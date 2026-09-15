#!/usr/bin/env python3
"""Single command-line entry point for the Wilson-Cowan gPINN study."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def cfg(path):
    from src.utils import load_config
    return load_config(path)


def call_main(fn, argv):
    old=sys.argv
    try:
        sys.argv=[old[0]]+list(argv)
        return fn()
    finally:
        sys.argv=old


def train_defaults(c, continuity=True):
    d=c['data']; p=c['physics']; s=c['selector']; t=c['training']; o=c['outputs']
    out=o['final'] if continuity else o['no_continuity']
    a=['--data_dir',d['processed_dir'],'--output_dir',out,'--subject_prefix',d['subject_prefix'],
       '--chunks',*map(str,d['chunks']),'--num_chunks_for_scaling',str(len(d['chunks'])),
       '--fps',str(d['fps']),'--tau',str(p['tau']),'--l_fov',str(d['fov_mm']),'--chunk_duration',str(d['chunk_duration_s']),
       '--nested_folds',str(s['nested_folds']),'--bootstrap_reps',str(s['bootstrap_reps']),'--bootstrap_block_frames',str(s['bootstrap_block_frames']),
       '--min_valid_fraction',str(s['min_valid_fraction']),'--max_standardized_condition',str(s['max_standardized_condition']),
       '--min_logit_improvement_pct',str(s['min_logit_improvement_pct']),'--min_pde_improvement_pct',str(s['min_pde_improvement_pct']),
       '--min_positive_fold_fraction',str(s['min_positive_fold_fraction']),'--max_smoothing_relative_range',str(s['max_smoothing_relative_range']),
       '--n_data_pool',str(t['n_data_pool']),'--n_val',str(t['n_val']),'--n_colloc_pool',str(t['n_colloc_pool']),'--n_deriv_pool',str(t['n_deriv_pool']),
       '--data_batch',str(t['data_batch']),'--colloc_batch',str(t['colloc_batch']),'--deriv_batch',str(t['deriv_batch']),
       '--stageA_steps',str(t['stageA_steps']),'--stageB_steps',str(t['stageB_steps']),'--lr_stageA',str(t['lr_stageA']),'--lr_stageB',str(t['lr_stageB']),
       '--lbfgs_steps',str(t['lbfgs_steps']),'--lambda_physics',str(t['lambda_physics']),'--lambda_deriv',str(t['lambda_deriv']),'--mu_grad',str(t['mu_grad']),
       '--lambda_cont',str(t['lambda_cont'] if continuity else 0.0),'--continuity_batch',str(t['continuity_batch']),'--seed',str(t['seed'])]
    return a


def verify(config_path):
    import numpy as np, torch
    from src import training
    from src.model import gPINN_WilsonCowan
    c=cfg(config_path); checks={}
    checks['config_20_chunks']=len(c['data']['chunks'])==20
    checks['config_shape']=tuple(c['data']['working_shape'])==(900,64,64)
    checks['continuity_lambda']=abs(c['training']['lambda_cont']-0.05)<1e-12
    checks['selector_available']=all(hasattr(training,n) for n in ['compute_pseudo_fields','select_model','FixedPhysicsGPINN'])
    torch.manual_seed(0)
    m=gPINN_WilsonCowan(ff_dim=4, hidden_neurons=8, hidden_layers=2, fourier_seed=0)
    x=torch.zeros((3,3),dtype=torch.float32)
    with torch.no_grad(): y=m.forward_E(x)
    checks['synthetic_model_forward_shape']=tuple(y.shape)==(3,1)
    ok=all(checks.values())
    print(json.dumps({'ok':ok,'checks':checks},indent=2))
    return 0 if ok else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config.yaml')
    sp=p.add_subparsers(dest='command',required=True)
    sp.add_parser('verify')
    sp.add_parser('env')
    vd=sp.add_parser('validate-data'); vd.add_argument('extra',nargs=argparse.REMAINDER)
    tr=sp.add_parser('train'); tr.add_argument('--no-continuity',action='store_true'); tr.add_argument('extra',nargs=argparse.REMAINDER)
    sy=sp.add_parser('synthetic'); sy.add_argument('action',choices=['generate','baseline','plot','oracle','staged','pseudoderiv','sweep','aggregate-sweep','selector']); sy.add_argument('extra',nargs=argparse.REMAINDER)
    dg=sp.add_parser('diagnostics'); dg.add_argument('action',choices=['independent','continuity','aggregate-continuity','derivatives','d-bound','regression','uncertainty']); dg.add_argument('extra',nargs=argparse.REMAINDER)
    rp=sp.add_parser('report'); rp.add_argument('action',choices=['extract','tables','figures','final-figures','all']); rp.add_argument('extra',nargs=argparse.REMAINDER)
    a=p.parse_args(); c=cfg(a.config)
    if a.command=='verify': return verify(a.config)
    if a.command=='env':
        from src.utils import check_environment; print(json.dumps(check_environment(),indent=2)); return 0
    if a.command=='validate-data':
        from src.utils import validate_processed_data
        rows=validate_processed_data(c['data']['processed_dir'],c['data']['subject_prefix'],len(c['data']['chunks']),tuple(c['data']['working_shape']))
        print(json.dumps(rows,indent=2)); return 0 if rows and all(r['ok'] for r in rows) else 1
    if a.command=='train':
        from src import training
        args=train_defaults(c,not a.no_continuity)+a.extra
        return call_main(training.main,args)
    if a.command=='synthetic':
        from src import synthetic
        fn={'generate':synthetic.generate,'baseline':synthetic.baseline_inverse,'plot':synthetic.plot_baseline,'oracle':synthetic.oracle,'staged':synthetic.staged,'pseudoderiv':synthetic.pseudoderiv,'sweep':synthetic.sweep,'aggregate-sweep':synthetic.aggregate_sweep,'selector':synthetic.selector}[a.action]
        return fn(a.extra)
    if a.command=='diagnostics':
        from src import diagnostics
        fn={'independent':diagnostics.independent_validation,'continuity':diagnostics.continuity,'aggregate-continuity':diagnostics.aggregate_continuity,'derivatives':diagnostics.derivative_diagnostic,'d-bound':diagnostics.d_bound_audit,'regression':diagnostics.regression_audit,'uncertainty':diagnostics.uncertainty_audit}[a.action]
        return fn(a.extra)
    if a.command=='report':
        from src import reporting
        if a.action!='all':
            fn={'extract':reporting.extract,'tables':reporting.tables,'figures':reporting.figures,'final-figures':reporting.final_figures}[a.action]
            return fn(a.extra)
        out=c['outputs']['publication']; final=c['outputs']['final']; noc=c['outputs']['no_continuity']; cont=c['outputs']['continuity']; synth=c['outputs']['synthetic']
        reporting.extract(['--project_root','.', '--final_dir',final,'--no_cont_dir',noc,'--continuity_dir',cont,'--synthetic_dir',synth,'--output_dir',out])
        reporting.tables(['--tables_dir',str(Path(out)/'tables')])
        common=['--project_root','.', '--master_dir',str(Path(out)/'master'),'--final_run_dir',final,'--data_dir',c['data']['processed_dir'],'--subject_prefix',c['data']['subject_prefix'],'--representative_chunk',str(c['reporting']['representative_chunk']),'--output_dir',str(Path(out)/'figures'),'--dpi',str(c['reporting']['dpi']),'--scatter_points',str(c['reporting']['scatter_points']),'--seed',str(c['training']['seed'])]
        reporting.figures(common); reporting.final_figures(common)
        return 0

if __name__=='__main__': raise SystemExit(main())
