#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=bgan-mnist
#
## output files
#SBATCH --output=exp_results/output/output-%j.log
#SBATCH --error=exp_results/output/output-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=0-8:00:00
#SBATCH --mem=100gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p gpu 
#SBATCH -N 2
#SBATCH -n 2

stdbuf -o0 ./run_bgan.py \
        --data_path $DATADIR/mnist \
        --dataset mnist \
        --out_dir exp_results \
        --gf_dim 64 --df_dim 64 \
        --disc_lr 0.0007 --enc_lr 0.001 --gen_lr 0.001 \
        --num_gen 5 --num_enc 1 --num_disc 1 --num_mcmc 1 \
        --train_iter 20000 \
        --n_save 500 --save_samples --evaluate_latent \
        --batch_size 256 \
        --d_update_threshold 1.0 --d_update_decay_steps 500,1000,2000,3000 --d_update_decay 0.025 \
        --d_update_bound 0.9 --lr_decay 0.0005  --e_optimize_iter 0 \
        --random_seed 1  
