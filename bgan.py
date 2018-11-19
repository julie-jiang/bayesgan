import sys
import numpy as np
import tensorflow as tf

from collections import OrderedDict, defaultdict

from bgan_util import AttributeDict

from dcgan_ops import *

DISC, GEN, ENC = "discriminator", "generator", "encoder"

def conv_out_size(size, stride):
    co = int(math.ceil(size / float(stride)))
    return co

def kernel_sizer(size, stride):
    ko = int(math.ceil(size / float(stride)))
    if ko % 2 == 0:
        ko += 1
    return ko


class BDCGAN(object):

    def __init__(self, x_dim, z_dim, dataset_size, batch_size=64, gf_dim=64, df_dim=64, 
                 prior_std=1.0, J=1, M=1, eta=2e-4, num_layers=4,
                 alpha=0.01, optimizer='adam', wasserstein=False, 
                 ml=False, J_d=1, J_e=1):


        assert len(x_dim) == 3, "invalid image dims"
        c_dim = x_dim[2]
        self.is_grayscale = (c_dim == 1)
        self.optimizer = optimizer.lower()
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        
        self.K = 2 # fake and real classes
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.ef_dim = df_dim # TODO
        self.c_dim = c_dim
        
        # Bayes
        self.prior_std = prior_std
        self.num_gen = J
        self.num_disc = J_d 
        self.num_enc = J_e
        self.num_mcmc = M
        self.eta = eta
        self.alpha = alpha
        # ML
        self.ml = ml
        if self.ml:
            assert self.num_gen == 1 and self.num_disc == 1 and \
                   self.num_enc == 1 and self.num_mcmc == 1, \
                   "invalid settings for ML training"

        self.noise_std = np.sqrt(2 * self.alpha * self.eta)

        def get_strides(num_layers, num_pool):
            interval = int(math.floor(num_layers / float(num_pool)))
            strides = np.array([1] * num_layers)
            strides[0:interval * num_pool:interval] = 2
            return strides

        self.num_pool = 4
        self.max_num_dfs = 512
        self.gen_strides = get_strides(num_layers, self.num_pool)
        self.disc_strides = self.gen_strides
        self.enc_strides = self.gen_strides
        num_dfs = np.cumprod(np.array([self.df_dim] + list(self.disc_strides)))[:-1]
        num_dfs[num_dfs >= self.max_num_dfs] = self.max_num_dfs # memory
        self.num_dfs = list(num_dfs)
        self.num_gfs = self.num_dfs[::-1]
        self.num_efs = self.num_dfs
        
        self.construct_from_hypers(
            gen_strides=self.gen_strides, 
            disc_strides=self.disc_strides,
            enc_strides=self.enc_strides,
            num_gfs=self.num_gfs, 
            num_dfs=self.num_dfs,
            num_efs=self.num_efs)
        
        self.build_bgan_graph()
    
    def construct_from_hypers(self, gen_kernel_size=5, gen_strides=[2, 2, 2, 2],
                              disc_kernel_size=5, disc_strides=[2, 2, 2, 2],
                              enc_kernel_size=5, enc_strides=[2, 2, 2, 2],
                              num_dfs=None, num_gfs=None, num_efs=None):

        
        self.d_batch_norm = AttributeDict(
            [("d_bn%i" % dbn_i, batch_norm(name='d_bn%i' % dbn_i)) \
             for dbn_i in range(len(disc_strides))])
        self.sup_d_batch_norm = AttributeDict(
            [("sd_bn%i" % dbn_i, batch_norm(name='sup_d_bn%i' % dbn_i)) \
             for dbn_i in range(5)])
        self.g_batch_norm = AttributeDict(
            [("g_bn%i" % gbn_i, batch_norm(name='g_bn%i' % gbn_i)) \
             for gbn_i in range(len(gen_strides))])
        self.e_batch_norm = AttributeDict(
            [("e_bn%i" % ebn_i, batch_norm(name="e_bn%i" % ebn_i)) \
             for ebn_i in range(len(enc_strides))])
    
        if num_dfs is None:
            num_dfs = [self.df_dim, self.df_dim * 2, self.df_dim * 4, self.df_dim * 8]
            
        if num_gfs is None:
            num_gfs = [self.gf_dim * 8, self.gf_dim * 4, self.gf_dim * 2, self.gf_dim]
        
        if num_efs is None:
            num_efs = [self.ef_dim * (2 ** i) for i in range(4)]
        
        assert len(gen_strides) == len(num_gfs), "invalid hypers!"
        assert len(disc_strides) == len(num_dfs), "invalid hypers!"
        assert len(enc_strides) == len(num_efs), "invalid hypers!"

        s_h, s_w = self.x_dim[0], self.x_dim[1]
        ks = gen_kernel_size
        self.gen_output_dims = OrderedDict()
        self.gen_weight_dims = OrderedDict()
        num_gfs = num_gfs + [self.c_dim]
        self.gen_kernel_sizes = [ks]
        for layer in range(len(gen_strides))[::-1]:
            self.gen_output_dims["g_h%i_out" % (layer + 1)] = (s_h, s_w)
            assert gen_strides[layer] <= 2, "invalid stride"
            assert ks % 2 == 1, "invalid kernel size"
            self.gen_weight_dims["g_h%i_W" % (layer + 1)] = \
                (ks, ks, num_gfs[layer + 1], num_gfs[layer])
            self.gen_weight_dims["g_h%i_b" % (layer + 1)] = (num_gfs[layer + 1],)
            s_h = conv_out_size(s_h, gen_strides[layer])
            s_w = conv_out_size(s_w, gen_strides[layer])
            ks = kernel_sizer(ks, gen_strides[layer])
            self.gen_kernel_sizes.append(ks)


        self.gen_weight_dims.update(OrderedDict(
            [("g_h0_lin_W", (self.z_dim, num_gfs[0] * s_h * s_w)),
             ("g_h0_lin_b", (num_gfs[0] * s_h * s_w,))]))
        self.gen_output_dims["g_h0_out"] = (s_h, s_w)

        self.disc_weight_dims = OrderedDict()
        s_h, s_w = self.x_dim[0], self.x_dim[1]
        num_dfs = [self.c_dim] + num_dfs
        ks = disc_kernel_size
        self.disc_kernel_sizes = [ks]
        for layer in range(len(disc_strides)):
            assert disc_strides[layer] <= 2, "invalid stride"
            assert ks % 2 == 1, "invalid kernel size"
            self.disc_weight_dims["d_h%i_W" % layer] = \
                (ks, ks, num_dfs[layer], num_dfs[layer + 1])
            self.disc_weight_dims["d_h%i_b" % layer] = (num_dfs[layer + 1],)
            s_h = conv_out_size(s_h, disc_strides[layer])
            s_w = conv_out_size(s_w, disc_strides[layer])
            ks = kernel_sizer(ks, disc_strides[layer])
            self.disc_kernel_sizes.append(ks)
        self.disc_weight_dims.update(OrderedDict(
            [("d_h_enc_lin_W", (self.z_dim, num_dfs[-1])),
             ("d_h_enc_lin_b", (num_dfs[-1],)),
             ("d_h0_lin_W", (num_dfs[-1] * s_h * s_w, num_dfs[-1])),
             ("d_h0_lin_b", (num_dfs[-1],)),
             ("d_h1_lin_W", (num_dfs[-1], num_dfs[-1])),
             ("d_h1_lin_b", (num_dfs[-1],)),
             ("d_h_out_lin_W", (num_dfs[-1], self.K)),
             ("d_h_out_lin_b", (self.K,))]))
        print("ADDED ONE MORE DISC LIN LAYER") 
        self.enc_weight_dims = OrderedDict()
        s_h, s_w = self.x_dim[0], self.x_dim[1]
        num_efs = [self.c_dim] + num_efs
        ks = enc_kernel_size
        self.enc_kernel_sizes = [ks]
        for layer in range(len(enc_strides)):
            assert enc_strides[layer] <= 2, "invalid strides"    
            assert ks % 2 == 1, "invalid kernel size"
            self.enc_weight_dims["e_h%i_W" % layer] = \
                (ks, ks, num_efs[layer], num_efs[layer + 1])
            self.enc_weight_dims["e_h%i_b" % layer] = (num_efs[layer + 1],)
            s_h = conv_out_size(s_h, enc_strides[layer])
            s_w = conv_out_size(s_w, enc_strides[layer])
            ks = kernel_sizer(ks, enc_strides[layer])
            self.enc_kernel_sizes.append(ks)
        
        self.enc_weight_dims.update(OrderedDict(
            [("e_h_end_lin_W", (num_efs[-1] * s_h * s_w, num_efs[-1])),
             ("e_h_end_lin_b", (num_efs[-1],)),
             ("e_h_out_lin_W", (num_efs[-1], self.z_dim)),
             ("e_h_out_lin_b", (self.z_dim,))]))
        
        for k, v in self.gen_output_dims.items():
            print("%s: %s" % (k, v))
        print('****')
        for k, v in self.gen_weight_dims.items():
            print("%s: %s" % (k, v))
        print('****')
        for k, v in self.disc_weight_dims.items():
            print("%s: %s" % (k, v))
        print("*****")
        for k, v in self.enc_weight_dims.items():
            print("%s: %s" % (k, v))
        
    def _get_optimizer(self, lr):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        elif self.optimizer == 'sgd':
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.5)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")    

    def initialize_wgts(self, scope_str):

        if scope_str == GEN:
            weight_dims = self.gen_weight_dims
            numz = self.num_gen
        elif scope_str == DISC:
            weight_dims = self.disc_weight_dims
            numz = self.num_disc
        elif scope_str == ENC:
            weight_dims = self.enc_weight_dims
            numz = self.num_enc
        else:
            raise RuntimeError("invalid scope!")

        param_list = []
        with tf.variable_scope(scope_str) as scope:
            for zi in range(numz):
                for m in range(self.num_mcmc):
                    wgts_ = AttributeDict()
                    for name, shape in weight_dims.items():
                        wgts_[name] = tf.get_variable(
                            "%s_%04d_%04d" % (name, zi, m),
                            shape, 
                            initializer=tf.random_normal_initializer(stddev=0.02)) # TODO?
                    param_list.append(wgts_)

            return param_list
        

    def build_bgan_graph(self):
    
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + self.x_dim, name='real_images')

        self.z = tf.placeholder(
            tf.float32, [self.batch_size, self.z_dim, self.num_gen], name='z')
        self.z_sampler = tf.placeholder(
            tf.float32, [self.batch_size, self.z_dim], name='z_sampler')
        
        # initialize generator weights
        self.gen_param_list = self.initialize_wgts(GEN)
        self.disc_param_list = self.initialize_wgts(DISC)
        self.enc_param_list = self.initialize_wgts(ENC)

        ### build discrimitive losses and optimizers
        # prep optimizer args
        self.d_learning_rate = tf.placeholder(tf.float32, shape=[])
        
        # compile all disciminative weights
        t_vars = tf.trainable_variables()
        self.d_vars = []
        for di in range(self.num_disc):
            for m in range(self.num_mcmc):
                self.d_vars.append(
                    [var for var in t_vars \
                     if 'd_h' in var.name and "_%04d_%04d" % (di, m) in var.name])
        ### build disc losses and optimizers
        self.d_losses_reals, self.d_losses_fakes = [], []
        self.d_optims_reals, self.d_optims_fakes = [], []
        self.d_optims_adam_reals, self.d_optims_adam_fakes = [], []
        for di, disc_params in enumerate(self.disc_param_list):

            d_prior_loss = self.prior(disc_params, DISC)
            d_losses_reals_ = []
            for enc_params in self.enc_param_list:
                encoded_inputs = self.encoder(self.inputs, enc_params)
                d_probs, d_logits, _ = self.discriminator(
                    self.inputs, encoded_inputs, self.K, disc_params)

                constant_labels = np.zeros((self.batch_size, self.K))
                constant_labels[:, 1] = 1.0  # real
                d_loss_real_ = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=d_logits,
                        labels=tf.constant(constant_labels)))
                if not self.ml:
                    d_loss_real_ += d_prior_loss + self.noise(disc_params, DISC)
                d_losses_reals_.append(tf.reshape(d_loss_real_, [1]))

            d_loss_reals = tf.reduce_logsumexp(tf.concat(d_losses_reals_, 0))    
            self.d_losses_reals.append(d_loss_reals)
 
            d_losses_fakes_ = []   
            for gi, gen_params in enumerate(self.gen_param_list):
                z = self.z[:, :, gi % self.num_gen]
                d_probs_, d_logits_, _ = self.discriminator(
                    self.generator(z, gen_params), z, self.K, disc_params)
                constant_labels = np.zeros((self.batch_size, self.K))
                # class label indicating it came from generator, aka fake
                constant_labels[:, 0] = 1.0
                d_loss_fake_ = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=d_logits_,
                        labels=tf.constant(constant_labels)))
                if not self.ml:
                    d_loss_fake_ += d_prior_loss + self.noise(disc_params, DISC)
                d_losses_fakes_.append(tf.reshape(d_loss_fake_, [1]))        
            
            d_loss_fakes = tf.reduce_logsumexp(tf.concat(d_losses_fakes_, 0))        
            self.d_losses_fakes.append(d_loss_fakes)
            # TODO???
            """
            for d_loss_ in d_loss_reals:
                if not self.ml:
                    d_loss_ += d_prior_loss + self.noise(disc_params, DISC)
                d_losses.append(tf.reshape(d_loss_, [1]))

            for d_loss_ in d_loss_fakes:
                #d_loss_ = d_loss_real_ * float(self.num_gen) + d_loss_fake_ # why???
                if not self.ml:
                    d_loss_ += d_prior_loss + \
                               self.noise(disc_params, DISC)
                d_losses.append(tf.reshape(d_loss_, [1]))
            """
            d_opt = self._get_optimizer(self.d_learning_rate)
            d_opt_adam = tf.train.AdamOptimizer(
                learning_rate=self.d_learning_rate, beta1=0.5)

            self.d_optims_reals.append(
                d_opt.minimize(d_loss_reals, var_list=self.d_vars[di]))
            self.d_optims_adam_reals.append(
                d_opt_adam.minimize(d_loss_reals, var_list=self.d_vars[di]))

            self.d_optims_fakes.append(
                d_opt.minimize(d_loss_fakes, var_list=self.d_vars[di]))
            self.d_optims_adam_fakes.append(
                d_opt_adam.minimize(d_loss_fakes, var_list=self.d_vars[di]))

        print("compiled discriminator losses\nd loss reals %r\nd_loss fakes %r" %
              (self.d_losses_reals, self.d_losses_fakes))

        ### build encoder losses and optimizers
        self.e_learning_rate = tf.placeholder(tf.float32, shape=[])
        self.e_losses, self.e_optims, self.e_optims_adam = [], [], []
        self.e_vars = []
        for ei in range(self.num_enc):
            for m in range(self.num_mcmc):
                self.e_vars.append(
                    [var for var in t_vars \
                     if "e_h" in var.name and "_%04d_%04d" % (ei, m) in var.name])     
        for ei, enc_params in enumerate(self.enc_param_list):
            ei_losses = []
            encoded_inputs = self.encoder(self.inputs, enc_params)
            e_prior_loss = self.prior(enc_params, ENC)
            for disc_params in self.disc_param_list:
                d_probs, d_logits, d_features = self.discriminator(
                    self.inputs, encoded_inputs, self.K, disc_params)
                constant_labels = np.zeros((self.batch_size, self.K)) 
                constant_labels[:, 0] = 1.0 # want to make real input appear fake
                
                e_loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=d_logits,
                        labels=tf.constant(constant_labels)))
                if not self.ml:
                    e_loss_ += e_prior_loss + self.noise(enc_params, ENC)
                
                ei_losses.append(tf.reshape(e_loss_, [1]))
            e_loss = tf.reduce_logsumexp(tf.concat(ei_losses, 0)) 
            self.e_losses.append(e_loss)
            e_opt = self._get_optimizer(self.e_learning_rate)
            self.e_optims.append(
                e_opt.minimize(e_loss, var_list=self.e_vars[ei]))
            e_opt_adam = tf.train.AdamOptimizer(
                learning_rate=self.e_learning_rate, beta1=0.5)
            self.e_optims_adam.append(
                e_opt_adam.minimize(e_loss, var_list=self.e_vars[ei]))
            
        print("compiled encoder losses", self.e_losses)

        ### build generative losses and optimizers
        self.g_learning_rate = tf.placeholder(tf.float32, shape=[])
        self.g_vars = []
        for gi in range(self.num_gen):
            for m in range(self.num_mcmc):
                self.g_vars.append(
                    [var for var in t_vars \
                     if 'g_h' in var.name and "_%04d_%04d" % (gi, m) in var.name])

        self.g_losses, self.g_optims, self.g_optims_adam = [], [], []
        for gi, gen_params in enumerate(self.gen_param_list):

            gi_losses = []
            g_prior_loss = self.prior(gen_params, GEN)
            for disc_params in self.disc_param_list:
                z = self.z[:, :, gi % self.num_gen]
                d_probs_, d_logits_, d_features_fake = self.discriminator(
                    self.generator(z, gen_params), z, self.K, disc_params)
                # class label indicating that this fake is real
                constant_labels[:, 1] = 1.0
                g_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=d_logits_,
                        labels=tf.constant(constant_labels)))
                """
                if not self.ml:
                    g_disc_loss += g_prior_loss + self.noise(gen_params, GEN)
                gi_losses.append(tf.reshape(g_disc_loss, [1]))
                # Also why???
                """
                for enc_params in self.enc_param_list:
                    encoded_inputs = self.encoder(self.inputs, enc_params)
                    _, _, d_features_real = self.discriminator(
                        self.inputs, encoded_inputs, self.K, disc_params)
                    g_hub_loss = tf.reduce_mean(
                        huber_loss(d_features_real, d_features_fake))
                    
                    g_loss_ = g_disc_loss + g_hub_loss
                    if not self.ml:
                        g_loss_ += g_prior_loss + self.noise(gen_params, GEN)
                    gi_losses.append(tf.reshape(g_loss_, [1]))
                #"""         
            g_loss = tf.reduce_logsumexp(tf.concat(gi_losses, 0))
            self.g_losses.append(g_loss)
            g_opt = self._get_optimizer(self.g_learning_rate)
            self.g_optims.append(g_opt.minimize(g_loss, var_list=self.g_vars[gi]))
            g_opt_adam = tf.train.AdamOptimizer(
                learning_rate=self.g_learning_rate, beta1=0.5)
            self.g_optims_adam.append(
                g_opt_adam.minimize(g_loss, var_list=self.g_vars[gi]))

        print("compiled generator losses", self.g_losses)      

        ### build samplers
        print("Not setting train to false in samplers/reconstructers/encoders")
        # TODO: set train to false?
        self.gen_samplers = []
        for gen_params in self.gen_param_list:
            self.gen_samplers.append(
                self.generator(self.z_sampler, gen_params))
        
        self.reconstructers = OrderedDict({})
        for gi, gen_params in enumerate(self.gen_param_list):
            for ei, enc_params in enumerate(self.enc_param_list):
                self.reconstructers[(gi, ei)] = self.generator(
                    self.encoder(self.inputs, enc_params), 
                    gen_params)
        
        self.encoders = []
        for enc_params in self.enc_param_list:
            self.encoders.append(
                self.encoder(self.inputs, enc_params))    
        

    def discriminator(self, image, encoded_image, K, disc_params, train=True):

        with tf.variable_scope(DISC, reuse=tf.AUTO_REUSE) as scope:

            h = image
            layer = 0

            h = lrelu(conv2d(h,
                             self.disc_weight_dims["d_h%i_W" % layer][-1],
                             name='d_h%i_conv' % layer,
                             k_h=self.disc_kernel_sizes[layer], 
                             k_w=self.disc_kernel_sizes[layer],
                             d_h=self.disc_strides[layer], 
                             d_w=self.disc_strides[layer],
                             w=disc_params["d_h%i_W" % layer], 
                             biases=disc_params["d_h%i_b" % layer]))
            
            for layer in range(1, len(self.disc_strides)):
                h = lrelu(self.d_batch_norm["d_bn%i" % layer](
                    conv2d(h,
                           self.disc_weight_dims["d_h%i_W" % layer][-1],
                           name='d_h%i_conv' % layer,
                           k_h=self.disc_kernel_sizes[layer], 
                           k_w=self.disc_kernel_sizes[layer],
                           d_h=self.disc_strides[layer], 
                           d_w=self.disc_strides[layer],
                           w=disc_params["d_h%i_W" % layer], 
                           biases=disc_params["d_h%i_b" % layer]), 
                    train=train))
            
            h_enc = lrelu(linear(
                encoded_image,
                self.df_dim * 4, # not needed, and not correct anways
                "d_h_enc_lin",
                matrix=disc_params["d_h_enc_lin_W"],
                bias=disc_params["d_h_enc_lin_b"]))

            h = tf.reshape(h, [self.batch_size, -1])
    
            for layer in range(2): #TODO           
                h = lrelu(linear(
                    h,
                    self.df_dim * 4, 
                    "d_h%d_lin" % layer,
                    matrix=disc_params["d_h%d_lin_W" % layer], 
                    bias=disc_params["d_h%d_lin_b" % layer])) # for feature norm
                h += h_enc 

            h_out = linear(
                h, 
                K, 
                'd_h_out_lin',
                matrix=disc_params["d_h_out_lin_W"], 
                bias=disc_params["d_h_out_lin_b"])
            
            return tf.nn.softmax(h_out), h_out, h
    
    def encoder(self, image, enc_params, train=True):
        with tf.variable_scope(ENC, reuse=tf.AUTO_REUSE) as scope:
            h = image
            layer = 0
      
            h = lrelu(conv2d(h,
                             self.enc_weight_dims["e_h%i_W" % layer][-1],
                             name='e_h%i_conv' % layer,
                             k_h=self.enc_kernel_sizes[layer], 
                             k_w=self.enc_kernel_sizes[layer],
                             d_h=self.enc_strides[layer], 
                             d_w=self.enc_strides[layer],
                             w=enc_params["e_h%i_W" % layer], 
                             biases=enc_params["e_h%i_b" % layer]))
            
            for layer in range(1, len(self.enc_strides)):
                h = lrelu(self.e_batch_norm["e_bn%i" % layer](
                    conv2d(h,
                           self.enc_weight_dims["e_h%i_W" % layer][-1],
                           name='e_h%i_conv' % layer,
                           k_h=self.enc_kernel_sizes[layer], 
                           k_w=self.enc_kernel_sizes[layer],
                           d_h=self.enc_strides[layer], 
                           d_w=self.enc_strides[layer],
                           w=enc_params["e_h%i_W" % layer], 
                           biases=enc_params["e_h%i_b" % layer]), 
                    train=train))
        
            h_end = lrelu(linear(
                tf.reshape(h, [self.batch_size, -1]),
                self.ef_dim * 4,
                "e_h_end_lin",
                matrix=enc_params["e_h_end_lin_W"],
                bias=enc_params["e_h_end_lin_b"]))
            
            h_out = lrelu(linear(
                h_end,
                self.z_dim,
                "e_h_out_lin",
                matrix=enc_params["e_h_out_lin_W"],
                bias=enc_params["e_h_out_lin_b"]))
            
            return h_out
                        

    def generator(self, z, gen_params, train=True):

        with tf.variable_scope(GEN, reuse=tf.AUTO_REUSE) as scope:

            h = linear(z, self.gen_weight_dims["g_h0_lin_W"][-1], 'g_h0_lin',
                       matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b)
            h = tf.nn.relu(self.g_batch_norm["g_bn0"](h, train=train))

            h = tf.reshape(h, [self.batch_size, 
                               self.gen_output_dims["g_h0_out"][0],
                               self.gen_output_dims["g_h0_out"][1], -1])

            for layer in range(1, len(self.gen_strides) + 1):

                out_shape = [self.batch_size, 
                             self.gen_output_dims["g_h%i_out" % layer][0],
                             self.gen_output_dims["g_h%i_out" % layer][1], 
                             self.gen_weight_dims["g_h%i_W" % layer][-2]]

                h = deconv2d(h,
                             out_shape,
                             k_h=self.gen_kernel_sizes[layer - 1], 
                             k_w=self.gen_kernel_sizes[layer - 1],
                             d_h=self.gen_strides[layer - 1], 
                             d_w=self.gen_strides[layer - 1],
                             name='g_h%i' % layer,
                             w=gen_params["g_h%i_W" % layer], 
                             biases=gen_params["g_h%i_b" % layer])
                if layer < len(self.gen_strides):
                    h = tf.nn.relu(self.g_batch_norm["g_bn%i" % layer](h, train=train))

            return tf.nn.tanh(h)        
    
        
    def prior(self, params, scope_str):
        assert scope_str in [DISC, GEN, ENC], \
               "invalid scope!"
        with tf.variable_scope(scope_str) as scope:
            prior_loss = 0.0
            for var in params.values():
                nn = tf.divide(var, self.prior_std)
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))
                
        prior_loss /= self.dataset_size

        return prior_loss

    def noise(self, params, scope_str): 
        assert scope_str in [DISC, GEN, ENC], \
               "invalid scope!"

        with tf.variable_scope(scope_str) as scope:
            noise_loss = 0.0
            for name, var in params.items():
                noise_ = tf.contrib.distributions.Normal(
                    loc=0., scale=self.noise_std * tf.ones(var.get_shape()))
                noise_loss += tf.reduce_sum(var * noise_.sample())
        noise_loss /= self.dataset_size
        return noise_loss
