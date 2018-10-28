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
#SBATCH --mem=150gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p gpu 
#SBATCH -N 2
#SBATCH -n 2
stdbuf -o0 ./run_bgan.py \
        --data_path mnistdir --out_dir exp_results/mnistunsup \
        --gf_dim 64 --df_dim 64 --num_gen 2 --num_disc 1 --num_mcmc 2 \
        --train_iter 5000 --n_save 1000 --save_samples --save_weights \
        --prior_std 10
