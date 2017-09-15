from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def encoder(image, shared=0, reuse=False, name="encoder"):
    #always sharing the mu/sig layers to reduce and simplify outputs
    gf_dim = 64


    with tf.variable_scope(name):
        # vec is 1024
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        if shared == 5:

            out = image

        elif shared == 4:

            e1 = lrelu(batch_norm(conv2d(image, gf_dim, s=2, ks=5, name='g_e1_conv'), 'g_bn_e1'))
            out = e1
        elif shared == 3:

            e1 = lrelu(batch_norm(conv2d(image, gf_dim, s=2, ks=5, name='g_e1_conv'),'g_bn_e1'))
            e2 = lrelu(batch_norm(conv2d(e1, gf_dim * 2, ks=5, s=2, name='g_e2_conv'), 'g_bn_e2'))
            out = e2
        elif shared == 2:

            e1 = conv2d(image, gf_dim, s=2, ks=5, name='g_e1_conv')
            e2 = batch_norm(conv2d(lrelu(e1), gf_dim * 2, ks=5, s=2, name='g_e2_conv'), 'g_bn_e2')
            e3 = batch_norm(conv2d(lrelu(e2), gf_dim * 4, ks=8, s=2, name='g_e3_conv'), 'g_bn_e3')
            out = e3
        elif shared == 1 or shared == 0:
            e1 = conv2d(image, gf_dim, s=2, ks=5, name='g_e1_conv')
            e2 = batch_norm(conv2d(lrelu(e1), gf_dim * 2, ks=5, s=2, name='g_e2_conv'), 'g_bn_e2')
            e3 = batch_norm(conv2d(lrelu(e2), gf_dim * 4, ks=8, s=2, name='g_e3_conv'), 'g_bn_e3')
            e4 = batch_norm(conv2d(lrelu(e3), gf_dim * 8, ks=1, s=2, name='g_e4_conv'), 'g_bn_e4')
            out = e4
        else:
            raise Exception

        return out


def encoder_shared(image, shared=0, lastmult=2, reuse=False, name="encoder_shared"):
    gf_dim = 64
    #lastmult = 2
    #2 for lastsize 64
    #4 for lastsize 128
    #8 for lastzise 256
    #16 for lastsize 512
    s = None
    mu = None
    s2 = None
    mu2 = None
    s3 = None
    mu3 = None


    with tf.variable_scope(name):
        # vec is 1024
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        if shared == 5:
            e1 = conv2d(image, gf_dim, s=2, ks=5, name='g_e1_conv')
            e2 = batch_norm(conv2d(lrelu(e1), gf_dim * 2, ks=5, s=2, name='g_e2_conv'), 'g_bn_e2')
            e3 = batch_norm(conv2d(lrelu(e2), gf_dim * 4, ks=8, s=2, name='g_e3_conv'), 'g_bn_e3')
            e4 = batch_norm(conv2d(lrelu(e3), gf_dim * 8, ks=1, s=2, name='g_e4_conv'), 'g_bn_e4')
            s = 1e-6 + tf.nn.softplus(conv2d(lrelu(e4), int(gf_dim * lastmult), s=1, ks=1, name='g_s_conv'), 'g_bn_s')
            mu = conv2d(lrelu(e4), int(gf_dim * lastmult), s=1, ks=1, name='g_mu_conv')
            out = e4

        elif shared == 4:

            e2 = lrelu(batch_norm(conv2d(image, gf_dim * 2, ks=5, s=2, name='g_e2_conv'), 'g_bn_e2'))
            e3 = lrelu(batch_norm(conv2d(e2, gf_dim * 4, ks=8, s=1, padding='VALID', name='g_e3_conv'), 'g_bn_e3'))
            e4 = lrelu(batch_norm(conv2d(e3, gf_dim * 8, ks=1, s=1, padding='VALID', name='g_e4_conv'), 'g_bn_e4'))

            s = tf.nn.softplus(conv2d(e4,int(gf_dim * lastmult), s=1, ks=1, name='g_s_conv'))
            mu = conv2d(e4, int(gf_dim * lastmult), s=1, ks=1, name='g_mu_conv')

            #s2 = 1e-6 + tf.nn.softplus(conv2d(lrelu(e2), gf_dim * 4, s=1, ks=1, name='g_s2_conv'))
            #mu2 = conv2d(lrelu(e2), gf_dim * 4, s=1, ks=1, name='g_mu2_conv')

            #s3 = 1e-6 + tf.nn.softplus(conv2d(lrelu(e3), gf_dim * 8, s=1, ks=1, name='g_s3_conv'))
            #mu3 = conv2d(lrelu(e3), gf_dim * 8, s = 1, ks = 1, name = 'g_mu3_conv')



            out = e4
        elif shared == 3:

            e3 = lrelu(batch_norm(conv2d(image, gf_dim * 4, ks=8, s=2, name='g_e3_conv'), 'g_bn_e3'))
            e4 = lrelu(batch_norm(conv2d(e3, gf_dim * 8, ks=1, s=1, padding='VALID', name='g_e4_conv'), 'g_bn_e4'))

            s = 1e-8 + tf.nn.softplus(conv2d(lrelu(e4), int(gf_dim * lastmult), s=1, ks=1, name='g_s_conv'), 'g_bn_s')
            mu = conv2d(lrelu(e4), int(gf_dim * lastmult), s=1, ks=1, name='g_mu_conv')

            out = e4
        elif shared == 2:

            e4 = batch_norm(conv2d(lrelu(image), gf_dim * 8, ks=1, s=2, name='g_e4_conv'), 'g_bn_e4')
            s = tf.nn.softplus(conv2d(lrelu(e4), int(gf_dim * lastmult), s=1, ks=1, name='g_s_conv'), 'g_bn_s')
            mu = conv2d(lrelu(e4), int(gf_dim * lastmult), s=1, ks=1, name='g_mu_conv')
            out = e4
        elif shared == 1 or shared == 0:
            s = 1e-6 + tf.nn.softplus(conv2d(lrelu(image), int(gf_dim * lastmult), s=1, ks=1, name='g_s_conv'), 'g_bn_s')
            mu = conv2d(lrelu(image), int(gf_dim * lastmult), s=1, ks=1, name='g_mu_conv')
            out = image
        else:
            raise Exception

        return out, s, mu, s2, mu2, s3, mu3


def decoder(vec, reuse=False,shared=0, name="decoder_shared"):

    gf_dim = 64

    with tf.variable_scope(name):
        # vec is 1024
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        if shared == 0:
            d1 = batch_norm(deconv2d(vec, gf_dim * 8, ks=4, name='g_d1'), 'g_bn_d1')
            d2 = batch_norm(deconv2d(lrelu(d1), gf_dim * 4, ks=4, name='g_d2'), 'g_bn_d2')
            d3 = batch_norm(deconv2d(lrelu(d2), gf_dim * 2, ks=4, name='g_d3'), 'g_bn_d3')
            d4 = batch_norm(deconv2d(lrelu(d3), gf_dim * 1, ks=4, name='g_d4'), 'g_bn_d4')
            d5 = deconv2d(lrelu(d4), 3, s=1, ks=1, name='g_d5')
            out = d5

        elif shared == 1:
            d2 = batch_norm(deconv2d(lrelu(vec), gf_dim * 4, ks=4, name='g_d2'), 'g_bn_d2')
            d3 = batch_norm(deconv2d(lrelu(d2), gf_dim * 2, ks=4, name='g_d3'), 'g_bn_d3')
            d4 = batch_norm(deconv2d(lrelu(d3), gf_dim * 1, ks=4, name='g_d4'), 'g_bn_d4')
            d5 = deconv2d(lrelu(d4), 3, s=1, ks=1, name='g_d5')

        elif shared == 2:
            d3 = batch_norm(deconv2d(lrelu(vec), gf_dim * 2, ks=4, name='g_d3'), 'g_bn_d3')
            d4 = batch_norm(deconv2d(lrelu(d3), gf_dim * 1, ks=4, name='g_d4'), 'g_bn_d4')
            d5 = deconv2d(lrelu(d4), 3, s=1, ks=1, name='g_d5')

        elif shared == 3:
            d4 = lrelu(batch_norm(deconv2d(vec, gf_dim * 1, ks=4, name='g_d4'), 'g_bn_d4'))
            d5 = deconv2d(d4, 3, s=1, ks=1, padding='VALID', name='g_d5')

        elif shared == 4:
            d5 = deconv2d(lrelu(vec), 3, s=1, ks=1, name='g_d5')
        elif shared == 5:
            d5 = vec

        else:
            raise Exception

        return tf.nn.tanh(d5)
        # return tf.nn.sigmoid(d5)


def decoder_shared(vec, z2=None, z3=None, shared=0, reuse=False, name="decoder"):
    gf_dim = 64

    with tf.variable_scope(name):
        # vec is 1024
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        # vec = tf.expand_dims(tf.expand_dims(vec, 1),1)

        if shared == 5:
            d1 = batch_norm(deconv2d(vec, gf_dim * 8, ks=4, name='g_d1'), 'g_bn_d1')
            d2 = batch_norm(deconv2d(lrelu(d1), gf_dim * 4, ks=4, name='g_d2'), 'g_bn_d2')
            d3 = batch_norm(deconv2d(lrelu(d2), gf_dim * 2, ks=4, name='g_d3'), 'g_bn_d3')
            d4 = batch_norm(deconv2d(lrelu(d3), gf_dim * 1, ks=4, name='g_d4'), 'g_bn_d4')
            d5 = deconv2d(lrelu(d4), 3, s=1, ks=1, name='g_d5')
            out = d5

        elif shared == 4:
            d1 = batch_norm(deconv2d(vec, gf_dim * 8, ks=4, name='g_d1'), 'g_bn_d1')
            d2 = batch_norm(deconv2d(lrelu(d1), gf_dim * 4, ks=4, name='g_d2'), 'g_bn_d2')
            d3 = batch_norm(deconv2d(lrelu(d2), gf_dim * 2, ks=4, name='g_d3'), 'g_bn_d3')
            d4 = batch_norm(deconv2d(lrelu(d3), gf_dim * 1, ks=4, name='g_d4'), 'g_bn_d4')
            out = d4

        elif shared == 3:
            d1 = lrelu(batch_norm(deconv2d(vec, gf_dim * 8, ks=4, padding='VALID', name='g_d1'), 'g_bn_d1'))
            #d1_c = tf.concat((d1,z3),3)
            #d2 = batch_norm(deconv2d(lrelu(d1_c), gf_dim * 4, ks=4, name='g_d2'), 'g_bn_d2')
            d2 = lrelu(batch_norm(deconv2d(d1, gf_dim * 4, ks=4, name='g_d2'), 'g_bn_d2'))
            #d2_c = tf.concat((d2,z2),3)
            d3 = lrelu(batch_norm(deconv2d(d2, gf_dim * 2, ks=4, name='g_d3'), 'g_bn_d3'))
            out = d3

        elif shared == 2:
            d1 = batch_norm(deconv2d(vec, gf_dim * 8, ks=4, name='g_d1'), 'g_bn_d1')
            d2 = batch_norm(deconv2d(lrelu(d1), gf_dim * 4, ks=4, name='g_d2'), 'g_bn_d2')
            out = d2

        elif shared == 1:
            d1 = batch_norm(deconv2d(vec, gf_dim * 8, ks=4, name='g_d1'), 'g_bn_d1')
            out = d1

        elif shared == 0:
            out = vec

        return out

def discriminator_svhn(image, reuse=False, name="discriminator_lenet"):

    df_dim = 64

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        #h1x = maxpool2d(tf.nn.relu(conv2d(image, df_dim, ks=5, s=2, name='d_h1_conv')))

        h1 = maxpool2d(tf.nn.relu(conv2d(image, df_dim, ks=5, s=1, name='d_h1_conv')))

        #h2 = maxpool2d(tf.nn.relu(conv2d(h1, df_dim * 4, ks=5, s=2, name='d_h2_conv')))

        h1_d = tf.nn.dropout(h1, .1)
        #h2_d = tf.nn.dropout(maxpool2d(tf.nn.relu(conv2d(h1_d, df_dim * 4, ks=5, s=2, name='d_h2_conv'))),.3)
        return h1, h1_d

def discriminator_svhn_shared(h1, h1_d, reuse=False, name="discriminator_lenet_shared"):

    df_dim = 64

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        #tf.get_variable_scope().reuse_variables()
        #reuse weights for dropout
        h2_d = tf.nn.dropout(maxpool2d(tf.nn.relu(conv2d(h1_d, df_dim * 2, ks=5, s=1, name='d_h2_conv'))), .3)
        h3_d = tf.nn.dropout(maxpool2d(tf.nn.relu(conv2d(h2_d, df_dim * 4, ks=5, s=1, name='d_h3_conv'))), .5)
        h4_d = tf.nn.dropout(maxpool2d(tf.nn.relu(conv2d(h3_d, df_dim * 8, ks=5, s=1, name='d_h4_conv'))), .5)
        #reassert reuse

        h5_d = conv2d(h4_d, 1024, ks=2, s=1, padding='VALID', name='d_h5_conv')
        h6_d = conv2d(h5_d, 1, ks=1, s=1, padding='VALID', name='d_h6_conv')
        out = tf.reshape(h6_d, [-1, 1], name='out')

        h5_cl = conv2d(h4_d, 1024, ks=2, s=1, padding='VALID', name='d_h5_cl_conv')
        h6_cl = conv2d(h5_cl, 10, ks=1, s=1, padding='VALID', name='d_h6_cl_conv')
        cl = tf.reshape(h6_cl, [-1, 10], name='cl')

        #for feature matching
        tf.get_variable_scope().reuse_variables()
        h2 = maxpool2d(tf.nn.relu(conv2d(h1, df_dim * 2, ks=5, s=2, name='d_h2_conv')))
        h3 = maxpool2d(tf.nn.relu(conv2d(h2, df_dim * 4, ks=5, s=2, name='d_h3_conv')))  # for feature matching
        h4 = maxpool2d(tf.nn.relu(conv2d(h3, df_dim * 8, ks=5, s=2, name='d_h4_conv')))
        feats = h4
        # h4 = conv2d(h3, 2, ks=2, s=1, name='d_h4_conv')

        #flat = tf.reshape(h3, [-1, 4096], name='flat')
        #fc1 = tf.nn.relu(linear(flat, 1024, 'd_fc1'))
        #out = linear(fc1, 1, 'd_fc_out')
        #cl1 = linear(fc1, 1024, 'd_cl1')
        #cl2 = linear(cl1, 10, 'd_cl2')

        return out, cl, feats


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
