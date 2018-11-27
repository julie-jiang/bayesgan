#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=bgan-cifar
#
## output files
#SBATCH --output=exp_results/output/output-%j.log
#SBATCH --error=exp_results/output/output-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=0-3:00:00
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
        --data_path $DATADIR/cifar \
        --dataset cifar \
        --out_dir exp_results \
        --gf_dim 64 --df_dim 64 \
        --disc_lr 0.001 --enc_lr 0.001 --gen_lr 0.001 \
        --num_gen 4 --num_enc 1 --num_disc 1 --num_mcmc 1 \
        --train_iter 15000 \
        --n_save 1000 --save_samples --evaluate_latent \
        --batch_size 256 
        #--prior_std 10 
#--ml 
#       --optimizer sgd 
#        --load_from "exp_results/mnistunsup/bgan_mnist_1542495198/model.ckpt-5000"
#        --data_path mnistdir --out_dir exp_results/mnistunsup \
#        --gf_dim 64 --df_dim 64 --num_gen 2 --num_disc 1 --num_mcmc 2 \
#        --train_iter 5000 --n_save 1000 --save_samples --save_weights \
#        --prior_std 10
