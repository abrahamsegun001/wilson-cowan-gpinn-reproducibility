# HPC and SLURM notes

The supplied shell scripts were developed on Digital Research Alliance of Canada infrastructure and contain Narval-specific module names, paths, GPU requests, and account directives. They are included to preserve the actual computational workflow.

Before using them on another account or cluster, review at least:

- `#SBATCH --account=...`
- GPU syntax and available GPU type
- Python/CUDA module versions
- the virtual environment activation path
- `~/scratch/gpinn_project` paths

The Python programs themselves are not tied to SLURM. For example, from the repository root you can run:

```bash
python train_gpinn_narval_final.py --data_dir data/processed --output_dir outputs/test_run --chunks 0
```

For the production manuscript configuration use the arguments in `job_gpinn_narval_final_cont005.sh`.
