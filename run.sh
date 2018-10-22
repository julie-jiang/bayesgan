#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=bgan
#
## output files
#SBATCH --output=exp_results/output/output-%j.log
#SBATCH --error=exp_results/output/output-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=0-7:00:00
#SBATCH --mem=50gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p gpu 
#SBATCH -N 2
#SBATCH -n 2
#./run_bgan.py --data_path mnistdir --out_dir exp_results/mnistunsup --train_iter 2000 --n_save 100 --save_samples --save_weights
./run_bgan_semi.py --data_path mnistdir --outdir exp_results/mnistsemi --train_iter 2000 --n_save 100 --save_samples --save_weights
