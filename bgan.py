import sys
import numpy as np
import tensorflow as tf

from collections import OrderedDict, defaultdict

from bgan_util import AttributeDict

from dcgan_ops import *

DISC, GEN, ENC = "disc", "gen", "enc"
FAKE_LABELS, REAL_LABELS = 1, 0

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
                 ml=False, J_d=1, J_e=1, d_learning_rate=0.001, 
                 g_learning_rate=0.001, e_learning_rate=0.001):
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
        
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate
        self.e_learning_rate = e_learning_rate
        
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
        
        self.construct_disc_from_hypers(disc_strides=self.disc_strides, 
                                        num_dfs=self.num_dfs)
        self.construct_gen_from_hypers(gen_strides=self.gen_strides, 
                                       num_gfs=self.num_gfs)
        self.construct_enc_from_hypers(enc_strides=self.enc_strides,
                                       num_efs=self.num_efs)
        
        self.build_bgan_graph()

    def construct_disc_from_hypers(self, disc_kernel_size=5, 
                                   disc_strides=[2, 2, 2, 2], num_dfs=None):
        self.d_batch_norm = AttributeDict(
            [("d_bn%i" % dbn_i, batch_norm(name='d_bn%i' % dbn_i)) \
             for dbn_i in range(len(disc_strides))])
        if num_dfs is None:
            num_dfs = [self.df_dim, self.df_dim * 2, self.df_dim * 4, self.df_dim * 8]


        assert len(disc_strides) == len(num_dfs), "invalid hypers!"

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
            [("d_h0_enc_lin_W", (self.z_dim, num_dfs[-1])),
             ("d_h0_enc_lin_b", (num_dfs[-1])),
             ("d_h1_enc_lin_W", (num_dfs[-1], num_dfs[-1])),
             ("d_h1_enc_lin_b", (num_dfs[-1],)),
             ("d_h2_enc_lin_W", (num_dfs[-1], num_dfs[-1])),
             ("d_h2_enc_lin_b", (num_dfs[-1],)),
             ("d_h_lin_W", (num_dfs[-1] * s_h * s_w, num_dfs[-1])),
             ("d_h_lin_b", (num_dfs[-1],)),
             ("d_h_out_lin_W", (num_dfs[-1], self.K)),
             ("d_h_out_lin_b", (self.K,))]))
        for k, v in self.disc_weight_dims.items():
            print("%s: %s" % (k, v))
        print("*****")

    def construct_gen_from_hypers(self, gen_kernel_size=5, 
                                  gen_strides=[2, 2, 2, 2], num_gfs=None):

        self.g_batch_norm = AttributeDict(
            [("g_bn%i" % gbn_i, batch_norm(name='g_bn%i' % gbn_i)) \
             for gbn_i in range(len(gen_strides))])
        if num_gfs is None:
            num_gfs = [self.gf_dim * 8, self.gf_dim * 4, self.gf_dim * 2, self.gf_dim]

        assert len(gen_strides) == len(num_gfs), "invalid hypers!"

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

        for k, v in self.gen_output_dims.items():
            print("%s: %s" % (k, v))
        print('****')
        for k, v in self.gen_weight_dims.items():
            print("%s: %s" % (k, v))
        print('****')

    def construct_enc_from_hypers(self, enc_kernel_size=5, 
                              enc_strides=[2, 2, 2, 2], num_efs=None):
        
        self.e_batch_norm = AttributeDict(
            [("e_bn%i" % ebn_i, batch_norm(name="e_bn%i" % ebn_i)) \
             for ebn_i in range(len(enc_strides))])
    
        if num_efs is None:
            num_efs = [self.ef_dim * (2 ** i) for i in range(4)]
        
        
        
        assert len(enc_strides) == len(num_efs), "invalid hypers!"

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
        
        for k, v in self.enc_weight_dims.items():
            print("%s: %s" % (k, v))
        print('****')
        
    def _get_optimizer(self, lr, use_adam=True):
        if self.optimizer == 'adam' or use_adam:
            return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        elif self.optimizer == 'sgd':
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.5)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")    
    
    def _compile_optimizers(self, scope_str, lr, loss, var_list):
        with tf.variable_scope(scope_str) as scope:
            opt_adam = self._get_optimizer(lr=lr, use_adam=True)
            opt_adam = opt_adam.minimize(loss, var_list=var_list)
            opt_user = self._get_optimizer(lr=lr, use_adam=False)
            opt_user = opt_user.minimize(loss, var_list=var_list)
            return opt_adam, opt_user
    
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
                            initializer=tf.glorot_uniform_initializer()) 
                    param_list.append(wgts_)

            return param_list
    def _get_vars(self, scope_str):
        if scope_str == GEN:
            prefix = "g"
            numz = self.num_gen
        elif scope_str == DISC:
            prefix = "d"
            numz = self.num_disc
        elif scope_str == ENC:
            prefix = "e"
            numz = self.num_enc
        else:
            raise RuntimeError("invalid scope!")
        var_list = []
        for i in range(numz):
            for m in range(self.num_mcmc):
                var_list.append(
                    [var for var in self.trainable_vars \
                    if "%s_h" % prefix in var.name and \
                       "_%04d_%04d" % (i, m) in var.name])
        return var_list

    def build_bgan_graph(self):
    
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + self.x_dim, name='real_images')

        self.z = tf.placeholder(
            tf.float32, [self.batch_size, self.z_dim, self.num_gen], name='z')
        self.z_sampler = tf.placeholder(
            tf.float32, [self.batch_size, self.z_dim], name='z_sampler')
        
        # initialize  weights
        self.gen_param_list = self.initialize_wgts(GEN)
        self.disc_param_list = self.initialize_wgts(DISC)
        self.enc_param_list = self.initialize_wgts(ENC)
        
        # get trainable vars
        self.trainable_vars = tf.trainable_variables()
        d_vars = self._get_vars(DISC)
        e_vars = self._get_vars(ENC)
        g_vars = self._get_vars(GEN)
        ### buil disc losses and optimizers
        self.opt_user_dict = {}
        self.opt_adam_dict = {}
        for m in [DISC, GEN, ENC]:
            self.opt_user_dict[m] = []
            self.opt_adam_dict[m] = []
        
        self.d_losses_reals, self.d_losses_fakes = [], []

        ### ALL_LOSSES
        disc_params = self.disc_param_list[0]
        enc_params = self.enc_param_list[0]

        d_prior_loss = self.prior(disc_params, DISC)
        d_losses_reals_ = []
        d_acc_reals_ = []
        
        encoded_inputs = self.encoder(self.inputs, enc_params)
        d_probs, d_logits, d_features_real = self.discriminator(
               self.inputs, encoded_inputs, self.K, disc_params)

        constant_labels = np.zeros((self.batch_size, self.K))
        constant_labels[:, REAL_LABELS] = 1.0  # real
        d_loss_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=d_logits,
                labels=tf.constant(constant_labels)))
        _, d_acc_real = tf.metrics.accuracy(
            tf.argmax(tf.constant(constant_labels), 1), tf.argmax(d_probs, 1))
        if not self.ml:
            d_loss_real_ = d_loss_real + d_prior_loss + self.noise(disc_params, DISC)
        d_losses_reals_.append(tf.reshape(d_loss_real_, [1]))
        d_acc_reals_.append(d_acc_real)

        d_loss_reals = tf.reduce_logsumexp(tf.concat(d_losses_reals_, 0))
        self.d_losses_reals.append(d_loss_reals)

        d_losses_fakes_ = []
        d_acc_fakes_ = []
        for gi, gen_params in enumerate(self.gen_param_list):
            z = self.z[:, :, gi % self.num_gen]
            d_probs_, d_logits_, _ = self.discriminator(
               self.generator(z, gen_params), z, self.K, disc_params)
            constant_labels = np.zeros((self.batch_size, self.K))
            # class label indicating it came from generator, aka fake
            constant_labels[:, FAKE_LABELS] = 1.0
            d_loss_fake_ = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=d_logits_,
                    labels=tf.constant(constant_labels)))
            d_loss_fake_ /= self.num_gen
            _, d_acc_fake = tf.metrics.accuracy(
                tf.argmax(tf.constant(constant_labels), 1), tf.argmax(d_probs_, 1))
            if not self.ml:
                d_loss_fake_ += d_prior_loss + self.noise(disc_params, DISC)
            d_losses_fakes_.append(tf.reshape(d_loss_fake_, [1]))
            d_acc_fakes_.append(d_acc_fake)
        d_loss_fakes = tf.reduce_logsumexp(tf.concat(d_losses_fakes_, 0))
        self.d_losses_fakes.append(d_loss_fakes)
            
        d_loss = tf.reduce_logsumexp(tf.concat(d_losses_reals_ + d_losses_fakes_, 0))
         
        d_opt_adam, d_opt_user = self._compile_optimizers(
            DISC, lr=self.d_learning_rate, loss=d_loss, var_list=d_vars[0])
        self.opt_user_dict[DISC].append(d_opt_user)
        self.opt_adam_dict[DISC].append(d_opt_adam)
            
        
        self.d_acc_reals = d_acc_reals_
        self.d_acc_fakes = d_acc_fakes_


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

        print("compiled discriminator losses\nd loss reals %r\nd_loss fakes %r" %
              (self.d_losses_reals, self.d_losses_fakes))

        ### compile e losses
        enc_params = self.enc_param_list[0]
        self.e_losses =  []
        ei_losses = []
        encoded_inputs = self.encoder(self.inputs, enc_params)
        e_prior_loss = self.prior(enc_params, ENC)
        d_probs, d_logits, d_features = self.discriminator(
            self.inputs, encoded_inputs, self.K, disc_params)
        constant_labels = np.zeros((self.batch_size, self.K)) 
        constant_labels[:, FAKE_LABELS] = 1.0 # want to make real input appear fake
               
        e_loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=d_logits,
                labels=tf.constant(constant_labels)))
        if not self.ml:
            e_loss_ += e_prior_loss + self.noise(enc_params, ENC)
            
        ei_losses.append(tf.reshape(e_loss_, [1]))
        e_loss = tf.reduce_logsumexp(tf.concat(ei_losses, 0)) 
        self.e_losses.append(e_loss)
        e_loss = self.e_losses[0]
        e_opt_adam, e_opt_user = self._compile_optimizers(
            ENC, lr=self.e_learning_rate, loss=e_loss, var_list=e_vars[0])
        self.opt_adam_dict[ENC].append(e_opt_adam)
        self.opt_user_dict[ENC].append(e_opt_user)
            
        print("compiled encoder losses", self.e_losses)

        ### compile g losses
        self.g_losses = []
        for gi, gen_params in enumerate(self.gen_param_list): 
            gi_losses = []
            g_prior_loss = self.prior(gen_params, GEN)
            z = self.z[:, :, gi % self.num_gen]
            d_probs_, d_logits_, d_features_fake = self.discriminator(
                self.generator(z, gen_params), z, self.K, disc_params)
            # class label indicating that this fake is real
            constant_labels = np.zeros((self.batch_size, self.K))
            constant_labels[:, REAL_LABELS] = 1.0
            g_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                   logits=d_logits_,
                   labels=tf.constant(constant_labels)))
            
            #g_disc_loss += 0.25 * tf.reduce_mean(huber_loss(d_features_real, d_features_fake))
            if not self.ml:
                g_disc_loss += g_prior_loss + self.noise(gen_params, GEN)
            gi_losses.append(tf.reshape(g_disc_loss, [1]))

            g_loss = tf.reduce_logsumexp(tf.concat(gi_losses, 0))
            self.g_losses.append(g_loss)

            
            g_opt_adam, g_opt_user = self._compile_optimizers(
                GEN, lr=self.g_learning_rate, loss=g_loss, var_list=g_vars[gi])
            
            
            self.opt_adam_dict[GEN].append(g_opt_adam)
            self.opt_user_dict[GEN].append(g_opt_user)
        print("compiled generator losses", self.g_losses)      

        ### build samplers
        print("Not setting train to false in samplers/reconstructers/encoders")
        # TODO: set train to false?
        self.gen_samplers = []
        for gen_params in self.gen_param_list:
            self.gen_samplers.append(
                self.generator(self.z_sampler, gen_params))
        
        self.reconstructers = OrderedDict({})
        self.recon_losses = []
        for gi, gen_params in enumerate(self.gen_param_list):
            for ei, enc_params in enumerate(self.enc_param_list):
                recon = self.generator(
                    self.encoder(self.inputs, enc_params), 
                    gen_params)
                self.reconstructers[(gi, ei)] = recon
                self.recon_losses.append(tf.losses.mean_squared_error(self.inputs, recon))
        
        
        self.encoders = []
        for enc_params in self.enc_param_list:
            self.encoders.append(
                self.encoder(self.inputs, enc_params))    
       
        self.discriminators = []
        for disc_param in self.disc_param_list:
            for enc_param in self.enc_param_list:
                _, d_logits, _ = self.discriminator(
                    self.inputs, 
                    self.encoder(self.inputs, enc_params),
                    self.K,
                    disc_params)
                self.discriminators.append(d_logits)

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
            
            h = lrelu(linear(
                tf.reshape(h, [self.batch_size, -1]),
                self.df_dim * 4,
                "d_h_lin",
                matrix=disc_params["d_h_lin_W"],
                bias=disc_params["d_h_lin_b"]))

            h_enc = encoded_image
            
            for layer in range(3): #TODO           
                h_enc = lrelu(linear(
                    h_enc,
                    self.disc_weight_dims["d_h%d_enc_lin_b" % layer], 
                    "d_h%d_enc_lin" % layer,
                    matrix=disc_params["d_h%d_enc_lin_W" % layer], 
                    bias=disc_params["d_h%d_enc_lin_b" % layer])) 
            
            h_out = linear(
                h + h_enc, 
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
            
            h_out = linear(
                h_end,
                self.z_dim,
                "e_h_out_lin",
                matrix=enc_params["e_h_out_lin_W"],
                bias=enc_params["e_h_out_lin_b"])
            
            return h_out #tf.nn.tanh(h_out)
                        

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



class BDCGAN_mnist(BDCGAN):
    def __init__(self, *args, **kwargs):
        self.x_disc_layers = 4
        self.z_disc_layers = 4
        super().__init__(*args, **kwargs)

    def construct_disc_from_hypers(self, **kwargs):
        self.disc_weight_dims = OrderedDict()

        x_dim = self.x_dim[0] * self.x_dim[1]
        x_disc_sizes = [x_dim, x_dim // 2, x_dim // 4, x_dim // 8, self.K]
       

        # x_dscrim
        for layer in range(self.x_disc_layers):
            self.disc_weight_dims["d_h%d_x_lin_W" % layer] = \
                (x_disc_sizes[layer], x_disc_sizes[layer + 1])
            self.disc_weight_dims["d_h%d_x_lin_b" % layer] = \
                (x_disc_sizes[layer + 1],)

        z_disc_sizes = [self.z_dim, self.z_dim * 2, self.z_dim * 4, self.z_dim, self.K]

        for layer in range(self.z_disc_layers):
            self.disc_weight_dims["d_h%d_z_lin_W" % layer] = \
                (z_disc_sizes[layer], z_disc_sizes[layer + 1])
            self.disc_weight_dims["d_h%d_z_lin_b" % layer] = \
                (z_disc_sizes[layer + 1],)
        

        self.disc_weight_dims["d_h_out_lin_W"] = (self.K, self.K)
        self.disc_weight_dims["d_h_out_lin_b"] = (self.K,)

        for k, v in self.disc_weight_dims.items():
            print("%s: %s" % (k, v))
        print("*****")

    def discriminator(self, image, encoded_image, K, disc_params, train=True):

        with tf.variable_scope(DISC, reuse=tf.AUTO_REUSE) as scope:

            h = tf.layers.flatten(image)

            for layer in range(self.x_disc_layers):
                h = lrelu(linear(h, 
                                 self.disc_weight_dims["d_h%d_x_lin_b" % layer],
                                 "d_h%d_x_lin" % layer,
                                 matrix=disc_params["d_h%d_x_lin_W" % layer],
                                 bias=disc_params["d_h%d_x_lin_b" % layer]))

            h_enc = encoded_image

            for layer in range(self.z_disc_layers):
                h_enc = lrelu(linear(h_enc, 
                                 self.disc_weight_dims["d_h%d_z_lin_b" % layer],
                                 "d_h%d_z_lin" % layer,
                                 matrix=disc_params["d_h%d_z_lin_W" % layer],
                                 bias=disc_params["d_h%d_z_lin_b" % layer]))
            
            h_out = linear(
                h + h_enc, 
                K, 
                'd_h_out_lin',
                matrix=disc_params["d_h_out_lin_W"], 
                bias=disc_params["d_h_out_lin_b"])
            
            return tf.nn.softmax(h_out), h_out, h
