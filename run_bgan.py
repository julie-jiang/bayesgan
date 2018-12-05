#!/usr/bin/env python3

import os
import sys
import argparse
import json
import time

import numpy as np
from math import ceil

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import slim

from bgan_util import AttributeDict
from bgan_util import print_images, MnistDataset, CelebDataset, Cifar10, SVHN, ImageNet
from bgan import BDCGAN, BDCGAN_mnist
from gan_plot import plot_losses, plot_latent_encodings

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score

def get_sess():
    if tf.get_default_session() is None:
        print("Creating new sess")
        tf.reset_default_graph()
        _SESSION = tf.InteractiveSession()
    else:
        print("Using old sess")
        _SESSION = tf.get_default_session()

    return _SESSION

def print_losses(name, losses):
    print("%s = %s" % 
         (name, ", ".join(["%.4f" % l for l in losses])))

def train_dcgan(dataset, args, dcgan, sess):

    print("Starting sess")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print("Starting training loop")
        
    num_train_iter = args.train_iter

    optimizer_dict = dcgan.opt_adam_dict
    
    
    lr_decay_rate = args.lr_decay
    num_disc = args.J_d
    saver = tf.train.Saver() 
    running_losses = {}
    
    for m in ["g", "e", "d_real", "d_fake", "recon"]:
        running_losses["%s_losses" % m] = np.empty(num_train_iter)
    running_losses["d_accuracy"] = np.empty(num_train_iter)
    d_update_threshold = args.d_update_threshold
    if args.d_update_decay_steps == "":
        d_update_decay_steps = []
    else:
        d_update_decay_steps = list(map(int, args.d_update_decay_steps.split(",")))
    base_lrs = {}
    base_lrs["g"] = args.gen_lr
    base_lrs["e"] = args.enc_lr
    base_lrs["d"] = args.disc_lr
    lrs = {}
    for k, v in base_lrs.items():
        lrs[k] = v
    d_lr = d_base_lr = args.disc_lr
    for train_iter in range(num_train_iter):
        for k, lr in base_lrs.items():
            if k != "d":
                continue
            lrs[k] = lr *  np.exp(
                -args.lr_decay * min(1.0, train_iter * args.batch_size / dataset.dataset_size))
        print('lrs', ', '.join("%s %.7f" % (k, lr) for k, lr in lrs.items()))
        if train_iter == 5000:
            print("Switching to user-specified optimizer")
            optimizer_dict = dcgan.opt_user_dict
        
        if (train_iter + 1) in d_update_decay_steps:
            d_update_threshold = max(d_update_threshold - args.d_update_decay, 
                                     args.d_update_bound)
            print("d update threshold:", d_update_threshold)
    
        image_batch, _ = dataset.next_batch(args.batch_size, class_id=None)       
        
        ### compute disc losses
        batch_z = np.random.uniform(-1, 1, [args.batch_size, args.z_dim, dcgan.num_gen])
        #np.random.normal(0, 1, [args.batch_size, args.z_dim, dcgan.num_gen])

        d_feed_dict = {dcgan.inputs: image_batch,
                       dcgan.z: batch_z,
                       dcgan.d_learning_rate: lrs["d"]}
        d_losses_reals, d_losses_fakes = sess.run(
            [dcgan.d_losses_reals, dcgan.d_losses_fakes], feed_dict=d_feed_dict)
        
        d_real_acc = sess.run(dcgan.d_acc_reals, feed_dict=d_feed_dict)
        d_fake_acc = sess.run(dcgan.d_acc_fakes, feed_dict=d_feed_dict)
        d_mean_acc = np.mean(np.concatenate((d_real_acc, d_fake_acc)))
        if (args.disc_skip_update and train_iter % 2) == 0 and d_mean_acc < d_update_threshold:
            sess.run(optimizer_dict["disc"], feed_dict=d_feed_dict)             
            d_updated = True
        else:
            d_updated = False
        ### compute encoder losses
        e_feed_dict = {dcgan.inputs: image_batch, dcgan.e_learning_rate: lrs["e"]}

        e_losses = sess.run(dcgan.e_losses, feed_dict=e_feed_dict)
        if train_iter + 1 > args.e_optimize_iter:
            sess.run(optimizer_dict["enc"], feed_dict=e_feed_dict)

        ### compute generative losses
        batch_z = np.random.uniform(-1, 1, [args.batch_size, args.z_dim, dcgan.num_gen])
        
        g_feed_dict = {dcgan.z: batch_z,
                       dcgan.inputs: image_batch, 
                       dcgan.g_learning_rate: lrs["g"]}
        g_losses = sess.run(dcgan.g_losses, feed_dict=g_feed_dict)
        sess.run(optimizer_dict["gen"], feed_dict=g_feed_dict)
        
        recon_losses = sess.run(dcgan.recon_losses, feed_dict={dcgan.inputs: image_batch})
        
        print("Iter %i" % train_iter)
        print_losses("Disc reals losses", d_losses_reals)
        print_losses("Disc fakes losses", d_losses_fakes)
        print_losses("Enc losses", e_losses)
        print_losses("Gen losses", g_losses)
        print_losses("Recon losses", recon_losses)        
        print_losses("Disc acc real", d_real_acc)
        print_losses("Disc acc fake", d_fake_acc)
        print("D mean acc", d_mean_acc)
        print("D updated", d_updated)
        running_losses["g_losses"][train_iter] = np.mean(g_losses)
        running_losses["e_losses"][train_iter] = np.mean(e_losses)
        running_losses["d_real_losses"][train_iter] = np.mean(d_losses_reals)
        running_losses["d_fake_losses"][train_iter] = np.mean(d_losses_fakes)
        running_losses["recon_losses"][train_iter] = np.mean(recon_losses)
        running_losses["d_accuracy"][train_iter] = d_mean_acc
        if train_iter + 1 == num_train_iter or \
           (train_iter > 0 and train_iter  % args.n_save == 0):

            """ print_losses("Raw Disc", raw_d_losses)
            print_losses("Raw Enc", raw_e_losses)
            print_losses("Raw Gen", raw_g_losses)
            """
            print("saving results and samples")

            results = {"disc_losses_reals": list(map(float, d_losses_reals)),
                       "disc_losses_fakes": list(map(float, d_losses_fakes)),
                       "enc_losses": list(map(float, e_losses)),
                       "gen_losses": list(map(float, g_losses)),
                       "timestamp": time.time()}
            res_path = os.path.join(args.out_dir, "results_%i.json" % train_iter)
            with open(res_path, 'w') as fp:
                json.dump(results, fp)
            
            if args.save_samples:
                for zi, gen_sampler in enumerate(dcgan.gen_samplers):
                    sampled_imgs = []
                    for _ in range(10):
                        z_sampler = np.random.uniform(
                            -1, 1, size=(args.batch_size, args.z_dim))
                        img = sess.run(
                            gen_sampler,
                            feed_dict={dcgan.z_sampler: z_sampler})
                        sampled_imgs.append(img)
                    sampled_imgs = np.concatenate(sampled_imgs)
                    print_images(
                        sampled_imgs, 
                        "B_DCGAN_g%i" % zi,
                        train_iter, 
                        directory=args.out_dir)
                
                #print_images(
                #    image_batch, "RAW", train_iter, directory=args.out_dir)
                evaluate_recon(sess, dcgan, args, dataset, train_iter) 
            if args.evaluate_latent: 
                all_latent_encodings = evaluate_latent(sess, dcgan, args, dataset)
                for ei, latent_encodings in enumerate(all_latent_encodings):
                    for r in range(1):
                        filename = "latent_encodings_e%d_r%d_%d.png" \
                                   % (ei, r, train_iter)
                        plot_latent_encodings(
                            latent_encodings, savename=os.path.join(args.out_dir, filename))
    

    save_path = saver.save(
        sess, 
        os.path.join(args.out_dir, "model.ckpt"))
    print("Model saved to %s" % save_path) 

    losses_file = os.path.join(args.out_dir, "running_losses.npz")          
    np.savez(losses_file, **running_losses)
    print("Saved running losses to", losses_file)
    
    plot_losses(running_losses, savename=os.path.join(args.out_dir, "losses_plot.png"))
    
    results = evaluate_classification(sess, dcgan, args, dataset)
    with open(os.path.join(args.out_dir, "classification.json"), "w") as fp:
        json.dump(results, fp)
    print("done")
def evaluate_recon(sess, dcgan, args, dataset, train_iter):
    for (gi, ei), recon in dcgan.reconstructers.items():
        for c in range(dataset.num_classes):
            inputs_c, _ = dataset.next_batch(args.batch_size, class_id=c)
            recons_c = sess.run(recon, feed_dict={dcgan.inputs: inputs_c})
            filename = "B_DCGAN_RECON_g%i_e%i_c%i" % (gi, ei, c)
            print_images(recons_c, filename, train_iter, directory=args.out_dir)
            # print_images(inputs_c, "RAW_c%i" % c, train_iter, directory=args.out_dir)
        
def evaluate_classification(sess, dcgan, args, dataset):
    def truncate_size(data):
        return data[:len(data) - len(data) % args.batch_size]
    results = {}
    Xtrain, ytrain = dataset.get_train_set()
    Xtest, ytest = dataset.get_test_set()
    Xtrain = truncate_size(Xtrain)
    ytrain = truncate_size(ytrain)
    Xtest = truncate_size(Xtest)
    ytest = truncate_size(ytest)
    for ei, encoder in enumerate(dcgan.encoders):
        Xtrain_feat = []
        
        for i in range(0, len(Xtrain), args.batch_size):
            Xbatch = Xtrain[i:i + args.batch_size]
            Xtrain_feat.extend(sess.run(encoder, feed_dict={dcgan.inputs: Xbatch}))
        Xtest_feat = []
        
        for i in range(0, len(Xtest), args.batch_size):
            Xbatch = Xtest[i:i + args.batch_size]
            Xtest_feat.extend(sess.run(encoder, feed_dict={dcgan.inputs: Xbatch}))
        Xtrain_feat, Xtest_feat = np.array(Xtrain_feat), np.array(Xtest_feat)
        acc = oneNN_classification(Xtrain_feat, ytrain, Xtest_feat, ytest)
        results["enc_%d" % ei] = acc
        print("Encoder %d 1NN classification accuracy: %f" % (ei, acc))
    
    for di, discriminator in enumerate(dcgan.discriminators):

        Xtrain_feat = []
        for i in range(0, len(Xtrain), args.batch_size):
            Xbatch = Xtrain[i:i + args.batch_size]
            Xtrain_feat.extend(sess.run(discriminator, feed_dict={dcgan.inputs: Xbatch}))
        Xtest_feat = []

        for i in range(0, len(Xtest), args.batch_size):
            Xbatch = Xtest[i:i + args.batch_size]
            Xtest_feat.extend(sess.run(discriminator, feed_dict={dcgan.inputs: Xbatch}))
        Xtrain_feat = np.array(Xtrain_feat)
        Xtest_feat = np.array(Xtest_feat)
        acc = oneNN_classification(Xtrain_feat, ytrain, Xtest_feat, ytest)
        results["disc_%d" % di] = acc
        print("Discriminator %d 1NN classification accuracy: %f" % (di, acc))
    
    return results
        

def oneNN_classification(Xtrain_feat, ytrain, Xtest_feat, ytest):
    clf = KNN(n_neighbors=1)
    clf.fit(Xtrain_feat, ytrain)
    ypred = clf.predict(Xtest_feat)

    acc = accuracy_score(ytest, ypred)
    return acc

def evaluate_latent(sess, dcgan, args, dataset, save_latent=False):
    all_latent_encodings = []
    for ei, encoder in enumerate(dcgan.encoders):
        latent_encodings = np.empty(
            (dataset.num_classes, args.batch_size, args.z_dim))
        for c in range(dataset.num_classes):
            inputs_c, _ = dataset.next_batch(args.batch_size, class_id=c)
            encodings_c = sess.run(
                encoder, 
                feed_dict={dcgan.inputs: inputs_c})
            latent_encodings[c] = encodings_c
        if save_latent:
            savepath = os.path.join(args.out_dir, "latent_encodings_%d.npy" % ei)
            np.save(savepath, latent_encodings)
            print("Latent encodings for encoder %d saved to %s" % (ei, savepath))
        all_latent_encodings.append(latent_encodings)
    return all_latent_encodings 
        
        
def b_dcgan(dataset, args):

    sess = get_sess()
    tf.set_random_seed(args.random_seed)

    if args.dataset == "mnist" and args.mnist_use_special_net:
        model = BDCGAN_mnist
    else:
        model = BDCGAN
    
    dcgan = model(dataset.x_dim, args.z_dim, 
                   dataset.dataset_size, batch_size=args.batch_size,
                   J=args.J, J_d=args.J_d, J_e=args.J_e, M=args.M, 
                   num_layers=args.num_layers,
                   optimizer=args.optimizer, gf_dim=args.gf_dim, 
                   df_dim=args.df_dim, prior_std=args.prior_std,
                   ml=(args.ml and args.J_e and args.J==1 and args.M==1 and args.J_d==1))
    
    if args.load_from is not None:
        saver = tf.train.Saver()
        saver.restore(sess, args.load_from)
        running_losses = os.path.join(args.out_dir, "running_losses.npz")
        running_losses = np.load(running_losses)
        plot_losses(savename=os.path.join(args.out_dir, "losses_plot.png"), **running_losses)
        evaluate_latent(sess, dcgan, args, dataset)
    else:
        train_dcgan(dataset, args, dcgan, sess)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN experiments')

    parser.add_argument('--out_dir',
                        type=str,
                        required=True,
                        help="location of outputs (root location, which exists)")

    parser.add_argument('--n_save',
                        type=int,
                        default=100,
                        help="every n_save iteration save samples and weights")
    
    parser.add_argument('--z_dim',
                        type=int,
                        default=100,
                        help='dim of z for generator')
    
    parser.add_argument('--gf_dim',
                        type=int,
                        default=64,
                        help='num of gen features')
    
    parser.add_argument('--df_dim',
                        type=int,
                        default=96,
                        help='num of disc features')
    
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='path to where the datasets live')

    parser.add_argument('--dataset',
                        type=str,
                        default="mnist",
                        help='datasate name mnist etc.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="minibatch size")

    parser.add_argument('--prior_std',
                        type=float,
                        default=1.0,
                        help="NN weight prior std.")

    parser.add_argument('--num_layers',
                        type=int,
                        default=4,
                        help="number of layers for G and D nets")

    parser.add_argument('--num_gen',
                        type=int,
                        dest="J",
                        default=1,
                        help="number of samples of z/generators")

    parser.add_argument('--num_disc',
                        type=int,
                        dest="J_d",
                        default=1,
                        help="number of discrimitor weight samples")

    parser.add_argument('--num_enc',
                        type=int,
                        dest="J_e",
                        default=1,
                        help="number of encoder samples")
   
    parser.add_argument('--num_mcmc',
                        type=int,
                        dest="M",
                        default=1,
                        help="number of MCMC NN weight samples per z")
    
    parser.add_argument('--N',
                        type=int,
                        default=128,
                        help="number of supervised data samples")

    parser.add_argument('--train_iter',
                        type=int,
                        default=50000,
                        help="number of training iterations")

    parser.add_argument('--wasserstein',
                        action="store_true",
                        help="wasserstein GAN")

    parser.add_argument('--ml',
                        action="store_true",
                        help="if specified, disable bayesian things")

    parser.add_argument('--save_samples',
                        action="store_true",
                        help="wether to save generated samples")

    parser.add_argument("--evaluate_latent",
                        action="store_true")

    parser.add_argument('--save_weights',
                        action="store_true",
                        help="wether to save weights")

    parser.add_argument('--random_seed',
                        type=int,
                        default=2222,
                        help="random seed")
    
    parser.add_argument('--gen_lr',
                        type=float,
                        default=0.001,
                        help="learning rate")

    parser.add_argument('--disc_lr',
                        type=float,
                        default=.0001)

    parser.add_argument('--enc_lr',
                        type=float,
                        default=.001)

    parser.add_argument('--lr_decay',
                        type=float,
                        default=3.0,
                        help="learning rate")

    parser.add_argument('--optimizer',
                        type=str,
                        default="sgd",
                        help="optimizer --- 'adam' or 'sgd'")
    
    parser.add_argument('--load_from',
                        type=str,
                        default=None)

    parser.add_argument('--d_update_threshold',
                        type=float,
                        default=1.)
    parser.add_argument('--d_update_decay_steps',
                        type=str,
                        default="")
    parser.add_argument('--d_update_decay',
                        type=float,
                        default=0.)
    parser.add_argument('--d_update_bound',
                        type=float,
                        default=1.)
    parser.add_argument('--e_optimize_iter',
                        type=int,
                        default=0)
    parser.add_argument('--mnist_use_special_net',
                        action="store_true")
    parser.add_argument('--disc_skip_update', action="store_true")

    args = parser.parse_args()
    print(args)
    # set seeds
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    
    if args.load_from:
        args.out_dir = os.path.dirname(args.load_from)
    else:
        if not os.path.exists(args.out_dir):
            print("Creating %s" % args.out_dir)
            os.makedirs(args.out_dir)
        args.out_dir = os.path.join(args.out_dir, "bgan_%s_%i" % (args.dataset, int(time.time())))
        os.makedirs(args.out_dir)

    import pprint
    with open(os.path.join(args.out_dir, "hypers.txt"), "w") as hf:
        hf.write("Hyper settings:\n")
        hf.write("%s\n" % (pprint.pformat(args.__dict__)))
        
    celeb_path = os.path.join(args.data_path, "celebA")
    cifar_path = os.path.join(args.data_path, "cifar-10-batches-py")
    svhn_path = os.path.join(args.data_path, "svhn")
    mnist_path = os.path.join(args.data_path, "mnist") # can leave empty, data will self-populate
    imagenet_path = os.path.join(args.data_path, args.dataset)
    #imagenet_path = os.path.join(args.data_path, "imagenet")

    if args.dataset == "mnist":
        dataset = MnistDataset(mnist_path)
    elif args.dataset == "celeb":
        dataset = CelebDataset(celeb_path)
    elif args.dataset == "cifar":
        dataset = Cifar10(cifar_path)
    elif args.dataset == "svhn":
        dataset = SVHN(svhn_path)
    elif "imagenet" in args.dataset:
        num_classes = int(args.dataset.split("_")[-1])
        dataset = ImageNet(imagenet_path, num_classes)
    else:
        raise RuntimeError("invalid dataset %s" % args.dataset)

    ### main call
    b_dcgan(dataset, args)
