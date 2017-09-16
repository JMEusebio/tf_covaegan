from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple

from module import *
from utils import *
import svhn_input_data
import mnist_input_data_test
import numpy


class cyclegan(object):
    def __init__(self, sess, args):

        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = 3
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.rep_size = args.rep_size
        self.sample_dir = args.sample_dir

        self.encoder = encoder
        self.encoder_shared = encoder_shared
        self.decoder = decoder
        self.decoder_shared = decoder_shared
        self.discriminator = discriminator_svhn
        self.discriminator_sh = discriminator_svhn_shared
        self.nzm = True

        self.dist = tf.contrib.distributions

        self.d_iter = 1
        self.g_iter = 1

        self.pernum = 10
        self.classes = 10
        self.labeledtarget = self.pernum * self.classes
        self.lbatchsize = min([64,self.labeledtarget])

        # generator hyper-parameters

        # VAE losses
        self.emu = 0
        self.esig = 1

        self.lbce = 1
        self.ltbce = 1
        self.lkld = .1

        # GAN losses

        self.ldrecon = .5
        self.ldtransl = .5

        # Classifier losses
        # for semi-supervised version
        self.lSSclass = 0
        self.lSTclass = 0
        self.lTTclass = 0
        self.lTSclasspseudo = 0  # labeled from classifier
        self.lTSclassreal = 0  # inherently labeled target image

        # Discriminator hyper-parameters
        self.ldfeats = 1

        self.ldsclass = 10
        self.ldstclass = 0
        self.ldtclass = 0

        # network variables # do not change, only configuration of 4,3 work right now
        self.encshared = 4
        self.decshared = 3

        # sizing for noise variables
        self.lastmult = 8 # * 2
        self.epsize = 64 * self.lastmult
        self.zdepth = 1

        # for storing max accuracies
        self.maxts = -1
        self.maxt = -1

        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc))

        self.target_name = "mnistn"
        self.source_name = "SVHN"

        self.color = True
        if self.target_name == "SVHN":
            self.loadedtargetdata = svhn_input_data.read_data_sets("./data/svhn", one_hot=True, color=self.color,
                                                                   lpernum=self.pernum)

        elif self.target_name == "mnist":
            self.loadedtargetdata = mnist_input_data_test.read_data_sets("./data/mnist", one_hot=True, color=self.color,
                                                                         noise=False, lpernum=self.pernum)

        else:
            self.loadedtargetdata = mnist_input_data_test.read_data_sets("./data/mnist", one_hot=True, color=self.color,
                                                                         noise=True, lpernum=self.pernum)


        if self.source_name == "SVHN":
            self.loadedsourcedata = svhn_input_data.read_data_sets("./data/svhn", one_hot=True, color=self.color,
                                                                   lpernum=self.pernum)

        elif self.source_name == "mnist":
            self.loadedsourcedata = mnist_input_data_test.read_data_sets("./data/mnist", one_hot=True, color=self.color,
                                                                         noise=False, lpernum=self.pernum)

        else:
            self.loadedsourcedata = mnist_input_data_test.read_data_sets("./data/mnist", one_hot=True, color=self.color,
                                                                         noise=True, lpernum=self.pernum)

        self.batch_testingtarget_x, self.batch_testingtarget_y = self.loadedtargetdata.test.testing_batch(1000)
        self.batch_testingsource_x, self.batch_testingsource_y = self.loadedsourcedata.test.testing_batch(1000)
        self.batch_samplingtarget_x, self.batch_samplingtarget_y = self.loadedtargetdata.test.testing_batch(64)
        self.batch_samplingsource_x, self.batch_samplingsource_y = self.loadedsourcedata.test.testing_batch(64)

        self.batch_labeledtargetexample_x, self.batch_labeledtargetexample_y = \
            self.loadedtargetdata.train.get_labeled()
        save_images(self.batch_labeledtargetexample_x, [self.classes, self.pernum],
                    './{}/test_sample.jpg'.format(args.sample_dir))


        self.num_batch = int(self.loadedtargetdata.train.num_examples / args.batch_size)
        self._build_model()
        self.saver = tf.train.Saver()

        self.savesample = True

    def _build_model(self):

        # placeholders for input during trianing
        self.real_S = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                     name='real_S_images')
        self.real_T = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                     name='real_T_images')

        self.labeled_T = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                     name='labeled_T_images')

        self.l_t_epsilon = tf.placeholder(tf.float32, [None,self.zdepth,self.zdepth, self.epsize], name='epsilon_t')
        self.l_s_epsilon = tf.placeholder(tf.float32, [None,self.zdepth,self.zdepth, self.epsize], name='epsilon_s')

        self.y_S = tf.placeholder(tf.float32, [None, 10], name='labels')
        self.y_T = tf.placeholder(tf.float32, [None, 10], name='labels_target')

        # BUILDING NETWORKS:
        # ENCODER NETWORK

        # intermediate values to exchange from unshared to shared in encoder
        self.e_t_logit = self.encoder(self.real_T, reuse=False, shared=self.encshared, name="encoder_t")
        self.e_s_logit = self.encoder(self.real_S, reuse=False, shared=self.encshared, name="encoder_s")

        self.e_lt_logit = self.encoder(self.labeled_T, reuse=False, shared=self.encshared, name="encoder_lt")

        self.e_t_logit_sh, self.e_t_sig_sh, self.e_t_mu_sh, \
            self.e_t_sig2_sh, self.e_t_mu2_sh, self.e_t_sig3_sh, self.e_t_mu3_sh = \
            encoder_shared(self.e_t_logit, lastmult=self.lastmult, reuse=False, shared=self.encshared, name="encoder_x")

        self.e_s_logit_sh, self.e_s_sig_sh, self.e_s_mu_sh,\
            self.e_s_sig2_sh, self.e_s_mu2_sh, self.e_s_sig3_sh, self.e_s_mu3_sh = \
            encoder_shared(self.e_s_logit, lastmult=self.lastmult, reuse=True, shared=self.encshared, name="encoder_x")

        self.e_lt_logit_sh, self.e_lt_sig_sh, self.e_lt_mu_sh, \
            self.e_lt_sig2_sh, self.e_lt_mu2_sh, self.e_lt_sig3_sh, self.e_lt_mu3_sh = \
            encoder_shared(self.e_lt_logit, lastmult=self.lastmult, reuse=True, shared=self.encshared, name="encoder_x")

        # SAMPLING

        self.l_s_epsilon = tf.random_normal(tf.shape(self.e_s_sig_sh), name='epsilon_s')  # [?, self.epsize]
        # self.e_s_std = tf.exp(.5 * self.e_s_sig_sh)  # [?, 30]
        self.e_s_std = self.e_s_sig_sh
        self.e_s_z = tf.add(self.e_s_mu_sh, tf.multiply(self.e_s_std, self.l_s_epsilon), name='e_s_z')

        self.l_t_epsilon = tf.random_normal(tf.shape(self.e_t_sig_sh), name='epsilon_t')  # [?, self.epsize]
        # self.e_t_std = tf.exp(.5 * self.e_t_sig_sh)  # [?, 30]
        self.e_t_std = self.e_t_sig_sh
        self.e_t_z = tf.add(self.e_t_mu_sh, tf.multiply(self.e_t_std, self.l_t_epsilon), name='e_t_z')

        self.l_lt_epsilon = tf.random_normal(tf.shape(self.e_lt_sig_sh), name='epsilon_lt')  # [?, self.epsize]
        # self.e_lt_std = tf.exp(.5 * self.e_lt_sig_sh)  # [?, 30]
        self.e_lt_std = self.e_lt_sig_sh
        self.e_lt_z = tf.add(self.e_lt_mu_sh, tf.multiply(self.e_lt_std, self.l_lt_epsilon), name='e_lt_z')

        # DECODER NETWORK
        # decoded images from z of real_images
        self.d_t_shared = self.decoder_shared(self.e_t_z,
                                              reuse=False, shared=self.decshared, name="decoder_x")
        self.d_s_shared = self.decoder_shared(self.e_s_z,
                                              reuse=True, shared=self.decshared, name="decoder_x")
        self.d_lt_shared = self.decoder_shared(self.e_lt_z,
                                              reuse=True, shared=self.decshared, name="decoder_x")

        self.ae_T_T = self.decoder(self.d_t_shared, shared=self.decshared, reuse=False, name="decoder_t")
        self.ae_T_S = self.decoder(self.d_t_shared, shared=self.decshared, reuse=False, name="decoder_s")
        self.ae_S_T = self.decoder(self.d_s_shared, shared=self.decshared, reuse=True, name="decoder_t")
        self.ae_S_S = self.decoder(self.d_s_shared, shared=self.decshared, reuse=True, name="decoder_s")

        self.ae_lT_S = self.decoder(self.d_lt_shared, shared=self.decshared, reuse=True, name="decoder_s")
        self.ae_lT_T = self.decoder(self.d_lt_shared, shared=self.decshared, reuse=True, name="decoder_t")

        # DISCRIMINATOR NETWORK
        # discriminators for generated images, inputs: real_images

        self.D_T_T, self.D_T_T_d = self.discriminator(self.ae_T_T, reuse=False, name="discriminatorT")
        self.D_T_S, self.D_T_S_d = self.discriminator(self.ae_T_S, reuse=False, name="discriminatorS")
        self.D_S_T, self.D_S_T_d = self.discriminator(self.ae_S_T, reuse=True, name="discriminatorT")
        self.D_S_S, self.D_S_S_d = self.discriminator(self.ae_S_S, reuse=True, name="discriminatorS")
        self.D_T_h, self.D_T_hd = self.discriminator(self.real_T, reuse=True, name="discriminatorT")
        self.D_S_h, self.D_S_hd = self.discriminator(self.real_S, reuse=True, name="discriminatorS")

        self.D_lT_S, self.D_lT_S_d = self.discriminator(self.ae_lT_S, reuse=True, name="discriminatorS")
        self.D_lT_T, self.D_lT_T_d = self.discriminator(self.ae_lT_T, reuse=True, name="discriminatorT")

        self.D_T_T_dom, self.D_T_T_class, self.D_T_T_feats = self.discriminator_sh(
            self.D_T_T, self.D_T_T_d, reuse=False, name="discriminatorX")
        self.D_T_S_dom, self.D_T_S_class, self.D_T_S_feats = self.discriminator_sh(
            self.D_T_S, self.D_T_S_d, reuse=True, name="discriminatorX")
        self.D_S_T_dom, self.D_S_T_class, self.D_S_T_feats = self.discriminator_sh(
            self.D_S_T, self.D_S_T_d, reuse=True, name="discriminatorX")
        self.D_S_S_dom, self.D_S_S_class, self.D_S_S_feats = self.discriminator_sh(
            self.D_S_S, self.D_S_S_d, reuse=True, name="discriminatorX")

        self.D_T_dom, self.D_T_cls, self.D_T_feats = self.discriminator_sh(self.D_T_h, self.D_T_hd, reuse=True,
                                                                              name="discriminatorX")
        self.D_S_dom, self.D_S_cls, self.D_S_feats = self.discriminator_sh(self.D_S_h, self.D_S_hd, reuse=True,
                                                                              name="discriminatorX")

        self.D_lT_S_dom, self.D_lT_S_class, self.D_lT_S_feats = self.discriminator_sh(
            self.D_lT_S, self.D_lT_S_d, reuse=True, name="discriminatorX")
        self.D_lT_T_dom, self.D_lT_T_class, self.D_lT_T_feats = self.discriminator_sh(
            self.D_lT_T, self.D_lT_T_d, reuse=True, name="discriminatorX")

        #elf.pseudolabel1 = tf.one_hot(tf.argmax(self.D_T_S_class,1),10)
        #self.pseudolabel2 = tf.nn.softmax(self.D_T_S_class, 1)

        self.pseudo_y_T = tf.placeholder(tf.float32, [None, 10], name='pseudo_labels')

        #LOSSES

        # losses for Generator

        # losses from discriminators
        self.ae_loss_T2T_d = self.ldrecon * sce_criterion(self.D_T_T_dom, tf.ones_like(self.D_T_T_dom))
        self.ae_loss_T2S_d = self.ldtransl * sce_criterion(self.D_T_S_dom, tf.ones_like(self.D_T_S_dom))
        self.ae_loss_S2T_d = self.ldtransl * sce_criterion(self.D_S_T_dom, tf.ones_like(self.D_S_T_dom))
        self.ae_loss_S2S_d = self.ldrecon * sce_criterion(self.D_S_S_dom, tf.ones_like(self.D_S_S_dom))

        if not self.lSTclass == 0:
            self.ae_loss_S_T_cl = self.lSTclass * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.D_S_T_class, labels=self.y_S))
        else:
            self.ae_loss_S_T_cl = 0

        if not self.lSSclass ==0:
            self.ae_loss_S_S_cl = self.lSSclass * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.D_S_S_class, labels=self.y_S))
        else:
            self.ae_loss_S_S_cl = 0

        if not self.lTTclass == 0:
            self.ae_loss_T_T_cl_labeled = self.lTTclass * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.D_lT_T_class, labels=self.y_T))
        else:
            self.ae_loss_T_T_cl_labeled = 0

        if not self.lTSclasspseudo == 0:
            self.ae_loss_T_S_cl_pseudo = self.lTSclasspseudo * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.D_T_S_class, labels=self.pseudo_y_T))
        else:
            self.ae_loss_T_S_cl_pseudo = 0

        if not self.lTSclassreal == 0:
            self.ae_loss_T_S_cl_labeled = self.lTSclassreal * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.D_lT_S_class, labels=self.y_T))
        else:
            self.ae_loss_T_S_cl_labeled = 0

        # losses from VAE


        #'''
        self.e_t_mu_sh_2 = tf.square(self.e_t_mu_sh)
        self.e_t_sig_sh_2 = tf.square(self.e_t_sig_sh)
        self.e_s_mu_sh_2 = tf.square(self.e_s_mu_sh)
        self.e_s_sig_sh_2 = tf.square(self.e_s_sig_sh)

        self.ae_loss_T_KLD = self.lkld * .5 * tf.reduce_mean(
            self.e_t_mu_sh_2 + self.e_t_sig_sh_2 - tf.log(self.e_t_sig_sh_2))

        self.ae_loss_S_KLD = self.lkld * .5 * tf.reduce_mean(
            self.e_s_mu_sh_2 + self.e_s_sig_sh_2 - tf.log(self.e_s_sig_sh_2))
        #'''
        #another way to do kl divergence:
        #self.ae_loss_T_KLD = self.lkld * tf.reduce_mean(-.5 * tf.reduce_sum(1. + self.e_t_sig_sh - tf.pow(self.e_t_mu_sh, 2)
        #                                         - tf.exp(self.e_t_sig_sh), reduction_indices=1))

        #self.ae_loss_S_KLD = self.lkld * tf.reduce_mean(-.5 * tf.reduce_sum(1. + self.e_s_sig_sh - tf.pow(self.e_s_mu_sh, 2)
        #                                         - tf.exp(self.e_s_sig_sh), reduction_indices=1))

        self.ae_loss_T_BCE = self.ltbce * tf.reduce_mean(tf.square(self.real_T - self.ae_T_T))
        self.ae_loss_S_BCE = self.lbce * tf.reduce_mean(tf.square(self.real_S - self.ae_S_S))

        self.ae_loss_T_vae = self.ae_loss_T_KLD + self.ae_loss_T_BCE
        self.ae_loss_S_vae = self.ae_loss_S_KLD + self.ae_loss_S_BCE

        # combining losses
        self.ae_loss_T = self.ae_loss_T_vae + self.ae_loss_T2T_d + self.ae_loss_S2T_d
        self.ae_loss_S = self.ae_loss_S_vae + self.ae_loss_S2S_d + self.ae_loss_T2S_d
        self.ae_loss = self.ae_loss_T + self.ae_loss_S + \
                       self.ae_loss_S_S_cl + self.ae_loss_S_T_cl + \
                       self.ae_loss_T_S_cl_pseudo + self.ae_loss_T_S_cl_labeled + \
                       self.ae_loss_T_T_cl_labeled

        self.ae_loss_vae = self.ae_loss_T_vae + self.ae_loss_S_vae

        # losses for Discriminator

        # to avoid repeated generation, placeholders for generated images
        self.fake_T_T_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                              name='fake_T_T_sample')
        self.fake_T_S_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                              name='fake_T_S_sample')

        self.fake_S_T_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                              name='fake_S_T_sample')
        self.fake_S_S_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                              name='fake_S_S_sample')

        self.fake_S_T_S_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                              name='fake_S_T_S_sample')

        self.fake_lT_S_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                              name='fake_lT_S_sample')

        # getting discriminator outputs for the various combinations of inputs

        self.D_T_T_h, self.D_T_T_hd  = self.discriminator(self.fake_T_T_sample, reuse=True,
                                                                             name="discriminatorT")
        self.D_T_S_h, self.D_T_S_hd = self.discriminator(self.fake_T_S_sample, reuse=True,
                                                                             name="discriminatorS")
        self.D_S_T_h, self.D_S_T_hd = self.discriminator(self.fake_S_T_sample, reuse=True,
                                                                             name="discriminatorT")
        self.D_S_S_h, self.D_S_S_hd = self.discriminator(self.fake_S_S_sample, reuse=True,
                                                                             name="discriminatorS")

        self.DlT_real_h, self.DlT_real_hd = self.discriminator(self.labeled_T, reuse=True, name="discriminatorT")

        self.D_T_T_dom_ex, self.D_T_T_cls_ex, self.D_T_T_feats_ex = self.discriminator_sh(self.D_T_T_h, self.D_T_T_hd, reuse=True,
                                                                             name="discriminatorX")
        self.D_T_S_dom_ex, self.D_T_S_cls_ex, self.D_T_S_feats_ex = self.discriminator_sh(self.D_T_S_h, self.D_T_S_hd, reuse=True,
                                                                             name="discriminatorX")
        self.D_S_T_dom_ex, self.D_S_T_cls_ex, self.D_S_T_feats_ex = self.discriminator_sh(self.D_S_T_h, self.D_S_T_hd, reuse=True,
                                                                             name="discriminatorX")
        self.D_S_S_dom_ex, self.D_S_S_cls_ex, self.D_S_S_feats_ex = self.discriminator_sh(self.D_S_S_h, self.D_S_S_hd, reuse=True,
                                                                             name="discriminatorX")

        self.DlT_dom_ex, self.DlT_cls_ex, self.DlT_feats_ex = self.discriminator_sh(self.DlT_real_h, self.DlT_real_hd, reuse=True, name="discriminatorX")

        # feature matching across domain translation
        self.d_T_loss_feat = self.ldfeats * mae_criterion(self.D_T_T_feats_ex - self.D_S_T_feats_ex,
                                                       tf.zeros_like(self.D_T_T_feats_ex))
        self.d_S_loss_feat = self.ldfeats * mae_criterion(self.D_S_S_feats_ex - self.D_T_S_feats_ex,
                                                        tf.zeros_like(self.D_S_S_feats_ex))

        # getting losses using discriminator inputs
        self.d_T_loss_real_dom = .5 * sce_criterion(self.D_T_dom, tf.ones_like(self.D_T_dom))
        self.d_T_T_loss_dom = .25 * sce_criterion(self.D_T_T_dom_ex, tf.zeros_like(self.D_T_T_dom_ex))
        self.d_S_T_loss_dom = .25 * sce_criterion(self.D_S_T_dom_ex, tf.zeros_like(self.D_S_T_dom_ex))

        self.d_S_loss_real_dom = .5 * sce_criterion(self.D_S_dom, tf.ones_like(self.D_S_dom))
        self.d_S_S_loss_dom = .25 * sce_criterion(self.D_S_S_dom_ex, tf.zeros_like(self.D_S_S_dom_ex))
        self.d_T_S_loss_dom = .25 * sce_criterion(self.D_T_S_dom_ex, tf.zeros_like(self.D_T_S_dom_ex))

        if self.ldtclass == 0:
            self.d_lT_loss_classifier = 0
        else:
            self.d_lT_loss_classifier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.DlT_cls_ex, labels=self.y_T))

        # this is really d_s_loss classifier
        if self.ldsclass == 0:
            self.d_S_loss_classifier = 0
        else:
            self.d_S_loss_classifier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.D_S_cls, labels=self.y_S))

        if self.ldstclass == 0:
            self.d_S_T_loss_classifier = 0
        else:
            self.d_S_T_loss_classifier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.D_S_T_cls_ex, labels=self.y_S))




        # combining losses for discriminator
        self.d_T_loss = self.d_T_loss_real_dom + self.d_T_T_loss_dom + self.d_S_T_loss_dom + \
                        self.ldstclass * self.d_S_T_loss_classifier + self.ldtclass* self.d_lT_loss_classifier + \
                        self.d_T_loss_feat

        self.d_S_loss = self.d_S_loss_real_dom + self.d_S_S_loss_dom + self.d_T_S_loss_dom + \
                        self.ldsclass * self.d_S_loss_classifier + self.d_S_loss_feat

        self.d_loss = self.d_T_loss + self.d_S_loss

        # TENSORBOARD SUMMARIES

        self.ae_loss_T2T_d_sum = tf.summary.scalar("ae_loss_T2T_d", self.ae_loss_T2T_d)
        self.ae_loss_S2T_d_sum = tf.summary.scalar("ae_loss_S2T_d", self.ae_loss_S2T_d)
        self.ae_loss_T_vae_sum = tf.summary.scalar("ae_loss_T_vae", self.ae_loss_T_vae)
        self.ae_loss_T_BCE_sum = tf.summary.scalar("ae_loss_T_BCE", self.ae_loss_T_BCE)
        self.ae_loss_T_KLD_sum = tf.summary.scalar("ae_loss_T_KLD", self.ae_loss_T_KLD)
        self.ae_T_sum = tf.summary.scalar("ae_loss_t", self.ae_loss_T)

        self.ae_loss_S2S_d_sum = tf.summary.scalar("ae_loss_S2S_d", self.ae_loss_S2S_d)
        self.ae_loss_T2S_d_sum = tf.summary.scalar("ae_loss_T2S_d", self.ae_loss_T2S_d)
        self.ae_loss_S_vae_sum = tf.summary.scalar("ae_loss_S_vae", self.ae_loss_S_vae)
        self.ae_loss_S_BCE_sum = tf.summary.scalar("ae_loss_S_BCE", self.ae_loss_S_BCE)
        self.ae_loss_S_KLD_sum = tf.summary.scalar("ae_loss_S_KLD", self.ae_loss_S_KLD)
        self.ae_S_sum = tf.summary.scalar("ae_loss_S", self.ae_loss_S)

        self.ae_loss_S_S_cl_sum = tf.summary.scalar("ae_loss_S_S_cl", self.ae_loss_S_S_cl)
        self.ae_loss_S_T_cl_sum = tf.summary.scalar("ae_loss_S_T_cl", self.ae_loss_S_T_cl)
        self.ae_loss_T_S_cl_labeled_sum = tf.summary.scalar("ae_loss_T_S_cl_labeled", self.ae_loss_T_S_cl_labeled)
        self.ae_loss_T_S_cl_pseudo_sum = tf.summary.scalar("ae_loss_T_S_cl_pseudo", self.ae_loss_T_S_cl_pseudo)
        self.ae_loss_T_T_cl_labeled_sum = tf.summary.scalar("ae_loss_T_T_cl", self.ae_loss_T_T_cl_labeled)

        self.ae_sum = tf.summary.scalar("ae_loss", self.ae_loss)

        self.d_T_loss_sum = tf.summary.scalar("d_T_loss", self.d_T_loss)
        self.d_T_loss_feat_sum = tf.summary.scalar("d_T_loss_feat", self.d_T_loss_feat)
        self.d_T_loss_real_sum = tf.summary.scalar("d_T_loss_real", self.d_T_loss_real_dom)
        self.d_T_T_loss_dom_sum = tf.summary.scalar("d_T_T_loss_dom", self.d_T_T_loss_dom)
        self.d_T_S_loss_dom_sum = tf.summary.scalar("d_T_S_loss_dom", self.d_T_S_loss_dom)

        self.d_S_loss_sum = tf.summary.scalar("d_S_loss", self.d_S_loss)
        self.d_S_loss_feat_sum = tf.summary.scalar("d_S_loss_feat", self.d_S_loss_feat)
        self.d_S_loss_real_sum = tf.summary.scalar("d_S_loss_real", self.d_S_loss_real_dom)
        self.d_S_S_loss_dom_sum = tf.summary.scalar("d_S_S_loss_dom", self.d_S_S_loss_dom)
        self.d_S_T_loss_dom_sum = tf.summary.scalar("d_S_T_loss_dom", self.d_S_T_loss_dom)

        self.d_lt_loss_real_classifier_sum = tf.summary.scalar("d_lT_real_classifier", self.d_lT_loss_classifier)
        self.d_S_loss_real_classifier_sum = tf.summary.scalar("d_S_real_classifier", self.d_S_loss_classifier)
        self.d_S_T_loss_real_classifier_sum = tf.summary.scalar("d_S_T_real_classifier", self.d_S_T_loss_classifier)

        self.ae_summary = tf.summary.merge(
            [self.ae_sum, self.ae_T_sum, self.ae_loss_T_vae_sum, self.ae_loss_T_BCE_sum, self.ae_loss_T_KLD_sum,
             self.ae_loss_S2S_d_sum, self.ae_loss_S2T_d_sum, self.ae_loss_T2S_d_sum, self.ae_loss_T2T_d_sum,
             self.ae_S_sum, self.ae_loss_S_vae_sum, self.ae_loss_S_BCE_sum, self.ae_loss_S_KLD_sum,
             self.ae_loss_S_S_cl_sum, self.ae_loss_S_T_cl_sum, self.ae_loss_T_T_cl_labeled_sum,
             self.ae_loss_T_S_cl_labeled_sum, self.ae_loss_T_S_cl_pseudo_sum
             ])

        self.d_summary = tf.summary.merge(
            [self.d_S_loss_sum, self.d_S_loss_real_sum, self.d_S_S_loss_dom_sum, self.d_T_S_loss_dom_sum,
             self.d_S_loss_real_classifier_sum,
             self.d_T_loss_sum, self.d_T_loss_real_sum, self.d_T_T_loss_dom_sum, self.d_S_T_loss_dom_sum,
             self.d_S_T_loss_real_classifier_sum, self.d_lt_loss_real_classifier_sum,
             self.d_S_loss_feat_sum, self.d_T_loss_feat_sum
             ]
        )

        # getting lists of variables to train specific sections
        t_vars = tf.trainable_variables()
        self.enc_S_vars = [var for var in t_vars if 'encoder_s' in var.name]
        self.dec_S_vars = [var for var in t_vars if 'decoder_s' in var.name]
        self.dsc_S_vars = [var for var in t_vars if 'discriminatorS' in var.name]

        self.enc_T_vars = [var for var in t_vars if 'encoder_t' in var.name]
        self.dec_T_vars = [var for var in t_vars if 'decoder_t' in var.name]
        self.dsc_T_vars = [var for var in t_vars if 'discriminatorT' in var.name]

        self.enc_x_vars = [var for var in t_vars if 'encoder_x' in var.name]
        self.dec_x_vars = [var for var in t_vars if 'decoder_x' in var.name]
        self.dsc_x_vars = [var for var in t_vars if 'discriminatorX' in var.name]

        self.train_S_enc_vars = self.enc_S_vars + self.dec_S_vars + self.dec_x_vars + self.enc_x_vars
        self.train_T_enc_vars = self.enc_T_vars + self.dec_T_vars + self.dec_x_vars + self.enc_x_vars
        self.train_dsc_S_vars = self.dsc_S_vars + self.dsc_x_vars
        self.train_dsc_T_vars = self.dsc_T_vars + self.dsc_x_vars

        self.train_dsc_vars = self.dsc_S_vars + self.dsc_x_vars + self.dsc_T_vars
        self.train_enc_vars = self.enc_S_vars + self.dec_S_vars +self.enc_T_vars + self.dec_T_vars + \
                              self.dec_x_vars + self.enc_x_vars

        self.ae_vars = self.enc_S_vars + self.dec_S_vars + self.dec_x_vars + self.enc_x_vars + \
                                               self.enc_T_vars + self.dec_T_vars



    def train(self, args):
        print("Training COVAEGAN")

        print("Initializing Optimizers")


        self.d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                         .minimize(self.d_loss, var_list=self.train_dsc_vars)

        self.ae_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                            .minimize(self.ae_loss, var_list=self.train_enc_vars)

        print("Initializing Variables")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.writer = tf.summary.FileWriter("C:/local/DCGAN/TFCNN/logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        self.load(args.checkpoint_dir)

        # loading pretrained ae
        self.loader_ae = tf.train.Saver(var_list=self.ae_vars)
        self.loadvars(args.checkpoint_dir, load_ae=True)

        self.checklistload = list()
        newvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for x in newvars:
            self.checklistload.append((x.name, self.sess.run(x)))

        self.test_classifier(args, load=False)

        print("Commencing Training")

        for epoch in xrange(args.epoch):
            batch_counter = 0
            while self.loadedtargetdata.train.epochs_completed < epoch+1:

                checklist2 = list()
                newvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                for x in newvars:
                    checklist2.append((x.name, self.sess.run(x)))

                batch_t_x, _ = self.loadedtargetdata.train.next_batch(self.batch_size)
                batch_s_x, batch_s_y = self.loadedsourcedata.train.next_batch(self.batch_size)

                self.batch_labeledtarget_x, self.batch_labeledtarget_y = \
                    self.loadedtargetdata.train.next_lbatch(self.lbatchsize)

                # Forward G network
                epsilon = numpy.random.normal(self.emu, self.esig,
                                              size=(self.batch_size, self.zdepth, self.zdepth, self.epsize))
                # for testing
                fake_T_T, fake_T_S = self.sess.run([self.ae_T_T, self.ae_T_S],
                                                   feed_dict={self.real_T: batch_t_x,self.l_t_epsilon: epsilon})

                fake_S_S, fake_S_T = self.sess.run([self.ae_S_S, self.ae_S_T],
                                                   feed_dict={self.real_S: batch_s_x, self.l_s_epsilon: epsilon,
                                                              self.l_t_epsilon: epsilon})

                for j in range(self.d_iter):

                    # Update D network
                    _, summary_str = self.sess.run([self.d_optim, self.d_summary],
                                                   feed_dict={self.real_S: batch_s_x, self.real_T: batch_t_x,
                                                              self.fake_T_S_sample: fake_T_S,
                                                              self.fake_S_S_sample: fake_S_S,
                                                              self.fake_T_T_sample: fake_T_T,
                                                              self.fake_S_T_sample: fake_S_T,
                                                              self.y_S: batch_s_y,
                                                              self.y_T: self.batch_labeledtarget_y,
                                                              self.labeled_T: self.batch_labeledtarget_x})
                    self.writer.add_summary(summary_str, counter)

                #pseudolabels for trianing
                fake_y_T = self.sess.run(self.pseudolabel1,
                                                   feed_dict={self.real_T: batch_t_x, self.l_t_epsilon: epsilon})

                for k in range(self.g_iter):
                    # Update G network
                    epsilon = numpy.random.normal(self.emu, self.esig,
                                                  size=(self.batch_size, self.zdepth, self.zdepth, self.epsize))

                    epsilont = numpy.random.normal(self.emu, self.esig,
                                                   size=(self.lbatchsize, self.zdepth, self.zdepth, self.epsize))

                    _, summary_str = self.sess.run([self.ae_optim, self.ae_summary],
                                                   feed_dict={self.real_S: batch_s_x, self.real_T: batch_t_x,
                                                              self.l_t_epsilon: epsilon, self.l_s_epsilon: epsilon,
                                                              self.y_S: batch_s_y, self.pseudo_y_T: fake_y_T,
                                                              self.y_T: self.batch_labeledtarget_y,
                                                              self.labeled_T: self.batch_labeledtarget_x,
                                                              self.l_lt_epsilon: epsilont,
                                                              })
                    self.writer.add_summary(summary_str, counter)

                counter += 1
                batch_counter +=1

                if np.mod(counter, 100) == 0:

                    print("Epoch: [%2d] [%4d:%4d/%4d] time: %4.4f" \
                          % (epoch, counter, batch_counter, self.num_batch, time.time() - start_time))


                if np.mod(counter, 100) == 0:
                    self.sample_model(args.sample_dir,epoch,counter)
                    self.test_classifier(args, load=False, counter=counter)

                if np.mod(counter, 200) == 0:
                    self.save(args.checkpoint_dir, counter)


    def trainVAE(self, args):
        print("Pre-Training VAE")


        print("Initializing Optimizers")

        self.ae_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.ae_loss_vae, var_list=self.ae_vars)

        print("Initializing Variables")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        counter = 1
        start_time = time.time()
        #loading whole checkpoint
        self.load(args.checkpoint_dir)

        print("Commencing Training VAE")

        for epoch in xrange(args.epoch):
            batch_counter = 0
            while self.loadedtargetdata.train.epochs_completed < epoch + 1:

                batch_t_x, _ = self.loadedtargetdata.train.next_batch(self.batch_size)
                batch_s_x, batch_s_y = self.loadedsourcedata.train.next_batch(self.batch_size)

                epsilon = numpy.random.normal(self.emu, self.esig, size=(self.batch_size, self.zdepth, self.zdepth, self.epsize))
                #epsilon2 = numpy.random.normal(self.emu, self.esig,
                #                               size=(self.batch_size, self.zdepth2, self.zdepth2, self.epsize2))
                #epsilon3 = numpy.random.normal(self.emu, self.esig,
                #                               size=(self.batch_size, self.zdepth3, self.zdepth3, self.epsize3))

                self.sess.run(self.ae_optim,
                              feed_dict={self.real_S: batch_s_x, self.real_T: batch_t_x,
                                         self.l_t_epsilon: epsilon, self.l_s_epsilon: epsilon})  #,
                                         # self.l_t_epsilon2: epsilon2, self.l_t_epsilon3: epsilon3,
                                         # self.l_s_epsilon2: epsilon2, self.l_s_epsilon3: epsilon3})
                counter += 1
                batch_counter += 1

                if np.mod(counter, 100) == 0:

                    print("Epoch: [%2d] [%4d:%4d/%4d] time: %4.4f" \
                          % (epoch, counter, batch_counter, self.num_batch, time.time() - start_time))

                if np.mod(counter, 100) == 0:
                    self.sample_model(args.sample_dir, epoch, counter)

                if np.mod(counter, 200) == 0:
                    self.savevars(args.checkpoint_dir, counter)

    def test_classifier(self, args, load=False, counter=0):


        if load:
            self.load(args.checkpoint_dir)


        tsize = 1000
        epsilon = numpy.random.normal(self.emu, self.esig,
                                      size=(tsize, self.zdepth, self.zdepth, self.epsize))

        input_S_x = self.batch_testingsource_x
        input_S_y = self.batch_testingsource_y

        input_T_x = self.batch_testingtarget_x
        input_T_y = self.batch_testingtarget_y

        #pred_S = self.D_S_S_class
        pred_S = self.D_S_cls
        pred_S_logit = tf.argmax(pred_S, 1)
        label_S_logit = tf.argmax(self.y_S, 1)
        label_S_predicted = pred_S_logit.eval({self.real_S: input_S_x, self.l_s_epsilon: epsilon})  # ,
                                               # self.l_s_epsilon2: epsilon2, self.l_s_epsilon3: epsilon3})
        label_S_true = label_S_logit.eval({self.y_S: input_S_y})
        acc_S = tf.equal(label_S_true, label_S_predicted)
        acc2_S = tf.reduce_mean(tf.cast(acc_S, "float"))
        accuracy_S = acc2_S.eval()
        print("AccuracyS:", accuracy_S)


        pred_SS = self.D_S_S_class
        pred_SS_logit = tf.argmax(pred_SS, 1)
        label_SS_logit = tf.argmax(self.y_S, 1)
        label_SS_predicted = pred_SS_logit.eval({self.real_S: input_S_x, self.l_s_epsilon: epsilon
                                                 # self.l_s_epsilon2: epsilon2, self.l_s_epsilon3: epsilon3
                                                 })
        label_SS_true = label_SS_logit.eval({self.y_S: input_S_y})
        acc_SS = tf.equal(label_SS_true, label_SS_predicted)
        acc2_SS = tf.reduce_mean(tf.cast(acc_SS, "float"))
        accuracy_SS = acc2_SS.eval()
        print("AccuracySS:", accuracy_SS)

        pred_ST = self.D_S_T_class
        pred_ST_logit = tf.argmax(pred_ST, 1)
        label_ST_logit = tf.argmax(self.y_S, 1)
        label_ST_predicted = pred_ST_logit.eval({self.real_S: input_S_x, self.l_s_epsilon: epsilon
                                                 #self.l_s_epsilon2: epsilon2, self.l_s_epsilon3: epsilon3
                                                 })
        label_ST_true = label_ST_logit.eval({self.y_S: input_S_y})
        acc_ST = tf.equal(label_ST_true, label_ST_predicted)
        acc2_ST = tf.reduce_mean(tf.cast(acc_ST, "float"))
        accuracy_ST = acc2_ST.eval()
        print("AccuracyST:", accuracy_ST)

        pred_T = self.D_T_cls
        pred_T_logit = tf.argmax(pred_T, 1)
        label_T_logit = tf.argmax(self.y_S, 1)
        label_T_predicted = pred_T_logit.eval({self.real_T: input_T_x})
        label_T_true = label_T_logit.eval({self.y_S: input_T_y})
        acc_T = tf.equal(label_T_true, label_T_predicted)
        acc2_T = tf.reduce_mean(tf.cast(acc_T, "float"))
        accuracy_T = acc2_T.eval()
        print("AccuracyT:", accuracy_T)

        pred_TT = self.D_T_T_class
        pred_TT_logit = tf.argmax(pred_TT, 1)
        label_TT_logit = tf.argmax(self.y_S, 1)
        label_TT_predicted = pred_TT_logit.eval({self.real_T: input_T_x,
                                                 self.l_t_epsilon: epsilon, self.l_s_epsilon: epsilon
                                                 })

        label_TT_true = label_TT_logit.eval({self.y_S: input_T_y})
        acc_TT = tf.equal(label_TT_true, label_TT_predicted)
        acc2_TT = tf.reduce_mean(tf.cast(acc_TT, "float"))
        accuracy_TT = acc2_TT.eval()
        print("AccuracyTT:", accuracy_TT)

        pred_TS = self.D_T_S_class
        pred_TS_logit = tf.argmax(pred_TS, 1)
        label_TS_logit = tf.argmax(self.y_S, 1)
        label_TS_predicted = pred_TS_logit.eval({self.real_T: input_T_x,
                                                 self.l_t_epsilon: epsilon, self.l_s_epsilon: epsilon
                                                 })
        label_TS_true = label_TS_logit.eval({self.y_S: input_T_y})
        acc_TS = tf.equal(label_TS_true, label_TS_predicted)
        acc2_TS = tf.reduce_mean(tf.cast(acc_TS, "float"))
        accuracy_TS = acc2_TS.eval()
        print("AccuracyT_S:", accuracy_TS)

        if accuracy_T> self.maxt:
            self.maxt = accuracy_T

        if accuracy_TS> self.maxts:
            self.maxts = accuracy_TS

        print('max_t_local=' + str(self.maxt))
        print('max_ts_local=' + str(self.maxts))

        if counter > 0:
            summary = tf.Summary(value=[tf.Summary.Value(tag="cl_S", simple_value=accuracy_S),
                                        tf.Summary.Value(tag="cl_SS", simple_value=accuracy_SS),
                                        tf.Summary.Value(tag="cl_ST", simple_value=accuracy_ST),
                                        tf.Summary.Value(tag="cl_T", simple_value=accuracy_T),
                                        tf.Summary.Value(tag="cl_TS", simple_value=accuracy_TS),
                                        tf.Summary.Value(tag="cl_TT", simple_value=accuracy_TT)])
            self.writer.add_summary(summary, counter)

        print("----------------------------------------------")

    def testfull(self):

        input_T_x = self.loadedtargetdata.test.images
        input_T_y = self.loadedtargetdata.test.labels

        tsize = input_T_x.shape[0]
        epsilon = numpy.random.normal(self.emu, self.esig,
                                      size=(tsize, self.zdepth, self.zdepth, self.epsize))
        pred_TS = self.D_T_S_class
        pred_TS_logit = tf.argmax(pred_TS, 1)
        label_TS_logit = tf.argmax(self.y_S, 1)
        label_TS_predicted = pred_TS_logit.eval({self.real_T: input_T_x,
                                                 self.l_t_epsilon: epsilon, self.l_s_epsilon: epsilon
                                                 })
        label_TS_true = label_TS_logit.eval({self.y_S: input_T_y})
        acc_TS = tf.equal(label_TS_true, label_TS_predicted)
        acc2_TS = tf.reduce_mean(tf.cast(acc_TS, "float"))
        accuracy_TS = acc2_TS.eval()
        print("max_ts_full:", accuracy_TS)

    def save(self, checkpoint_dir, step):
        model_name = "unit.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def savevars(self, checkpoint_dir, step):
        model_name = "pretrained.model"
        model_dir = "%s_%s" % (self.dataset_dir+"_pt", self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success reading {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    def loadvars(self, checkpoint_dir,load_class=False, load_ae=False):

        print(" [*] Reading checkpoint...")

        if load_class:
            model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
            checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.loader_class.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                print(" [*] Loaded pretrained classifier, Success reading {}".format(ckpt_name))
                return True
            else:
                print(" [*] Failed to find a checkpoint")
                return False
        if load_ae:
            model_dir = "%s_%s" % (self.dataset_dir+"_pt", self.image_size)
            checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.loader_ae.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                print(" [*] Loaded pretrained AE, Success reading {}".format(ckpt_name))
                return True
            else:
                print(" [*] Failed to find a checkpoint")
                return False

    def sample_model(self, sample_dir, epoch, idx):

        input_T_x = self.batch_samplingtarget_x
        input_S_x = self.batch_samplingsource_x

        size = 8

        if self.savesample:
            save_images(input_T_x, [size, size],
                        './{}/T_T___Sample.jpg'.format(sample_dir))
            save_images(input_T_x, [size, size],
                        './{}/T_S___Sample.jpg'.format(sample_dir))
            save_images(input_S_x, [size, size],
                        './{}/S_S___sample.jpg'.format(sample_dir))
            save_images(input_S_x, [size, size],
                        './{}/S_T___sample.jpg'.format(sample_dir))
            #save_images(input_S_x, [size, size],
                        #'./{}/S_T_S_sample.jpg'.format(sample_dir))
            self.savesample = False

        epsilon = numpy.random.normal(self.emu, self.esig,
                                      size=(self.batch_size, self.zdepth, self.zdepth, self.epsize))
        fake_T_T, fake_T_S, = self.sess.run(
            [self.ae_T_T, self.ae_T_S],
            feed_dict={self.real_T: input_T_x,
                       self.l_t_epsilon:epsilon, self.l_s_epsilon:epsilon
                       })

        fake_S_T, fake_S_S = self.sess.run(
            [self.ae_S_T, self.ae_S_S],
            feed_dict={self.real_S: input_S_x,
                       self.l_t_epsilon:epsilon, self.l_s_epsilon:epsilon
                       })

        save_images(fake_T_T, [size, size],
                    './{}/T_T___{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_S_S, [size, size],
                    './{}/S_S___{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

        save_images(fake_T_S, [size, size],
                    './{}/T_S___{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_S_T, [size, size],
                    './{}/S_T___{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

