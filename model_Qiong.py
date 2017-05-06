from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, 
         batch_size=64, test_num=100, 
         y_dim=None, z_dim=100, gf_dim=32, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3,
         checkpoint_dir=None, sample_dir=None,
         source_font=None, source_height=32, source_width=32,
         target_font=None, target_height=32, target_width=32,
         tv_penalty=0.0002, L1_penalty=50):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess

    self.batch_size = batch_size
    self.test_num = test_num
    self.tv_penalty = tv_penalty
    self.L1_penalty = L1_penalty

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')
    self.g_bn4 = batch_norm(name='g_bn4')
    self.g_bn5 = batch_norm(name='g_bn5')
    self.g_bn6 = batch_norm(name='g_bn6')
    self.g_bn7 = batch_norm(name='g_bn7')
    self.g_bn8 = batch_norm(name='g_bn8')
    self.g_bn9 = batch_norm(name='g_bn9')
    self.g_bn10 = batch_norm(name='g_bn10')
    self.g_bn11 = batch_norm(name='g_bn11')
    self.g_bn12 = batch_norm(name='g_bn12')
    self.g_bn13 = batch_norm(name='g_bn13')
    self.g_bn14 = batch_norm(name='g_bn14')


    self.checkpoint_dir = checkpoint_dir

    self.source_font = source_font
    self.source_height = source_height
    self.source_width = source_width
    self.target_font = target_font
    self.target_height = target_height
    self.target_width = target_width
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    source_dims = [self.source_height, self.source_width, self.c_dim]
    target_dims = [self.target_height, self.target_width, self.c_dim]

    self.source = tf.placeholder(
      tf.float32, [self.batch_size] + source_dims, name='source')
    self.target = tf.placeholder(
      tf.float32, [self.batch_size] + target_dims, name='target')  
    self.source_test = tf.placeholder(
      tf.float32, [self.test_num] + source_dims, name='source_test')       

    self.G = self.generator(self.source)
    self.sampler = self.sampler(self.source_test)

    self.source_target = tf.concat([self.source, self.target], 3)
    self.source_source = tf.concat([self.source, self.source], 3)
    self.source_G = tf.concat([self.source, self.G], 3)

    self.D_st_tf_logits, self.D_st_font_logits = self.discriminator(self.source_target)
    self.D_ss_tf_logits, self.D_ss_font_logits = self.discriminator(self.source_source, reuse=True)
    self.D_sG_tf_logits, self.D_sG_font_logits = self.discriminator(self.source_G, reuse=True)

    print(self.G)
    print(self.source)
    print(self.target)

    # self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    # self.d_sum = histogram_summary("d", self.D)
    # self.d__sum = histogram_summary("d_", self.D_)
    # self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    D_ones = tf.ones_like(self.D_st_tf_logits)
    D_zeros = tf.zeros_like(self.D_st_tf_logits)

    self.d_loss_st_tf = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_st_tf_logits, D_ones))
    self.d_loss_ss_tf = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_ss_tf_logits, D_ones))   
    self.d_loss_sG_tf = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_sG_tf_logits, D_zeros))   
    self.d_loss_tf = self.d_loss_st_tf * 0.5 + self.d_loss_ss_tf * 0.5 + self.d_loss_sG_tf

    self.d_loss_st_font = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_st_font_logits, D_ones))
    self.d_loss_ss_font = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_ss_font_logits, D_zeros))   
    self.d_loss_sG_font = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_sG_font_logits, D_zeros))
    self.d_loss_font = self.d_loss_st_font * 0.5 + self.d_loss_ss_font * 0.5 + self.d_loss_sG_font

    # self.d_loss = self.d_loss_tf + self.d_loss_font
    self.d_loss = self.d_loss_st_tf + self.d_loss_sG_tf

    self.tv_loss = total_variation_loss(self.G) * self.tv_penalty 
    self.L1_loss = tf.reduce_mean(tf.abs(self.G - self.source)) * self.L1_penalty

    self.g_loss_tf = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_sG_tf_logits, D_ones))    
    self.g_loss_font = tf.reduce_mean(  
      sigmoid_cross_entropy_with_logits(self.D_sG_font_logits, D_ones))

    # self.g_loss = self.g_loss_tf + self.g_loss_font + self.tv_loss + self.L1_loss
    self.g_loss = self.g_loss_tf + self.tv_loss + self.L1_loss


    # self.d_loss_real = tf.reduce_mean(
    #   sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    # self.d_loss_fake = tf.reduce_mean(
    #   sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    # self.g_loss = tf.reduce_mean(
    #   sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    # self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    # self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    # self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()
      
  def train(self, config):
    """Train DCGAN"""
    source_npy = read_font_data(self.source_font) \
                 .reshape([-1, self.source_height, self.source_width, self.c_dim])
    target_npy = read_font_data(self.target_font) \
                 .reshape([-1, self.target_height, self.target_width, self.c_dim])
    assert source_npy.shape[0] == target_npy.shape[0]

    source_test = source_npy[:self.test_num]
    target_test = target_npy[:self.test_num]
    source_train = source_npy[self.test_num:]
    target_train = target_npy[self.test_num:]

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    
    self.writer = SummaryWriter("./logs", self.sess.graph)

    self.g_sum = self.g_loss_sum
    self.d_sum = self.d_loss_sum
    
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      batch_idxs = min(source_train.shape[0], config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_source = source_train[idx*config.batch_size:(idx+1)*config.batch_size,:,:,:]
        batch_target = target_train[idx*config.batch_size:(idx+1)*config.batch_size,:,:,:]

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ self.source: batch_source, self.target: batch_target })
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.source: batch_source })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.source: batch_source })
        self.writer.add_summary(summary_str, counter)
        
        # errD_fake = self.d_loss_fake.eval({ self.source: batch_source })
        # errD_real = self.d_loss_real.eval({ self.target: batch_target })
        # errG = self.g_loss.eval({ self.source: batch_source })

        d_loss = self.d_loss.eval({ self.source: batch_source, self.target: batch_target })
        g_loss = self.g_loss.eval({ self.source: batch_source, self.target: batch_target })
        L1_loss = self.L1_loss.eval({self.source: batch_source, self.target: batch_target})
        tv_loss = self.tv_loss.eval({self.source: batch_source, self.target: batch_target})

        counter += 1
        log = "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f, L1_loss: %.4f, tv_loss: %.4f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, d_loss, g_loss, L1_loss, tv_loss)
        print(log)
            
        f = open('./{}/log.txt'.format(self.sample_dir), 'a')
        f.write(log)
        f.write('\n')
        f.close()  

        # save sample images
        if np.mod(counter, 2) == 1:
          try:
            samples = self.sess.run(self.sampler, feed_dict={self.source_test: source_test})
            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            save_images(samples, [manifold_h, manifold_w],
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))              
          except:
            print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      
      # whether the input is a character
      h4_tf = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin_tf')

      # whether the character is of target font
      h4_font = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin_font')

      return h4_tf, h4_font

  def generator(self, source):
    with tf.variable_scope("generator") as scope:
      # Using the architecture in https://arxiv.org/pdf/1611.06355.pdf

      # 256*256
      s_h, s_w = self.target_height, self.target_width
      # 128*128
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      # 64*64
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      # 32*32
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      # 16*16
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  
      # 8*8
      s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)  
      # 4*4
      s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)  
      # 2*2
      s_h128, s_w128 = conv_out_size_same(s_h64, 2), conv_out_size_same(s_w64, 2)  




      h0 = lrelu(self.g_bn0(conv2d(source, self.gf_dim, name='g_h0_conv'))) #128*128*32
      h1 = lrelu(self.g_bn1(conv2d(h0, self.gf_dim << 1, name='g_h1_conv'))) #64*64*64
      h2 = lrelu(self.g_bn2(conv2d(h1, self.gf_dim << 2, name='g_h2_conv'))) #32*32*128
      h3 = lrelu(self.g_bn3(conv2d(h2, self.gf_dim << 3, name='g_h3_conv'))) #16*16*256
      h4 = lrelu(self.g_bn4(conv2d(h3, self.gf_dim << 4, name='g_h4_conv'))) #8*8*512
      h5 = lrelu(self.g_bn5(conv2d(h4, self.gf_dim << 5, name='g_h5_conv'))) #4*4*1024
      h6 = lrelu(self.g_bn6(conv2d(h5, self.gf_dim << 6, name='g_h6_conv'))) #2*2*2056
      print(h6.shape)
      h7 = lrelu(self.g_bn7(linear(tf.reshape(h6, [self.batch_size, -1]), 4096, 'g_h7_lin')))
      print(h7.shape)
      h8 = linear(h7, 100, 'g_h8_lin')
      h8 = tf.reshape(h8, [self.batch_size, 1, 1, -1])
      print(h8.shape)
      h9 = lrelu(self.g_bn8(deconv2d(h8, 
        [self.batch_size, s_h128, s_w128, self.gf_dim << 7], k_h=4, k_w=4, d_h=self.target_height >> 7, d_w=self.target_height >> 7, name='g_h9_deconv')))
      h10 = lrelu(self.g_bn9(deconv2d(h9, 
        [self.batch_size, s_h64, s_w64, self.gf_dim << 6], k_h=4, k_w=4, name='g_h10_deconv')))
      h11 = lrelu(self.g_bn10(deconv2d(h10, 
        [self.batch_size, s_h32, s_w32, self.gf_dim << 5], k_h=4, k_w=4, name='g_h11_deconv')))
      h12 = lrelu(self.g_bn11(deconv2d(h11, 
        [self.batch_size, s_h16, s_w16, self.gf_dim << 4], k_h=4, k_w=4, name='g_h12_deconv')))      
      h13 = lrelu(self.g_bn12(deconv2d(h12, 
        [self.batch_size, s_h8, s_w8, self.gf_dim << 3], k_h=4, k_w=4, name='g_h13_deconv')))      
      h14 = lrelu(self.g_bn13(deconv2d(h13, 
        [self.batch_size, s_h4, s_w4, self.gf_dim << 2], k_h=4, k_w=4, name='g_h14_deconv')))      
      h15 = lrelu(self.g_bn14(deconv2d(h14, 
        [self.batch_size, s_h2, s_w2, self.gf_dim << 1], k_h=4, k_w=4, name='g_h15_deconv')))      
      h16 = lrelu(deconv2d(h15, 
        [self.batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, name='g_h16_deconv'))
      return tf.nn.sigmoid(h16)

  def sampler(self, source, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      # Using the architecture in https://arxiv.org/pdf/1611.06355.pdf

      # 256*256
      s_h, s_w = self.target_height, self.target_width
      # 128*128
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      # 64*64
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      # 32*32
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      # 16*16
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  
      # 8*8
      s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)  
      # 4*4
      s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)  
      # 2*2
      s_h128, s_w128 = conv_out_size_same(s_h64, 2), conv_out_size_same(s_w64, 2)

      h0 = lrelu(self.g_bn0(conv2d(source, self.gf_dim, name='g_h0_conv'))) #32*32
      h1 = lrelu(self.g_bn1(conv2d(h0, self.gf_dim << 1, name='g_h1_conv'))) #64*64
      h2 = lrelu(self.g_bn2(conv2d(h1, self.gf_dim << 2, name='g_h2_conv'))) #128*128
      h3 = lrelu(self.g_bn3(conv2d(h2, self.gf_dim << 3, name='g_h3_conv'))) #256*256
      h4 = lrelu(self.g_bn4(conv2d(h3, self.gf_dim << 4, name='g_h4_conv'))) #512*512
      h5 = lrelu(self.g_bn5(conv2d(h4, self.gf_dim << 5, name='g_h5_conv'))) #1028*1028
      h6 = lrelu(self.g_bn6(conv2d(h5, self.gf_dim << 6, name='g_h6_conv'))) #2056*2056
      h7 = lrelu(self.g_bn7(linear(tf.reshape(h6, [self.test_num, -1]), 4096, 'g_h7_lin')))
      h8 = linear(h7, 100, 'g_h8_lin')
      h8 = tf.reshape(h8, [self.test_num, 1, 1, -1])
      h9 = lrelu(self.g_bn8(deconv2d(h8, 
        [self.test_num, s_h128, s_w128, self.gf_dim << 7], k_h=4, k_w=4, d_h=self.target_height >> 7, d_w=self.target_height >> 7, name='g_h9_deconv')))
      h10 = lrelu(self.g_bn9(deconv2d(h9, 
        [self.test_num, s_h64, s_w64, self.gf_dim << 6], k_h=4, k_w=4, name='g_h10_deconv')))
      h11 = lrelu(self.g_bn10(deconv2d(h10, 
        [self.test_num, s_h32, s_w32, self.gf_dim << 5], k_h=4, k_w=4, name='g_h11_deconv')))
      h12 = lrelu(self.g_bn11(deconv2d(h11, 
        [self.test_num, s_h16, s_w16, self.gf_dim << 4], k_h=4, k_w=4, name='g_h12_deconv')))      
      h13 = lrelu(self.g_bn12(deconv2d(h12, 
        [self.test_num, s_h8, s_w8, self.gf_dim << 3], k_h=4, k_w=4, name='g_h13_deconv')))      
      h14 = lrelu(self.g_bn13(deconv2d(h13, 
        [self.test_num, s_h4, s_w4, self.gf_dim << 2], k_h=4, k_w=4, name='g_h14_deconv')))      
      h15 = lrelu(self.g_bn14(deconv2d(h14, 
        [self.test_num, s_h2, s_w2, self.gf_dim << 1], k_h=4, k_w=4, name='g_h15_deconv')))      
      h16 = lrelu(deconv2d(h15, 
        [self.test_num, s_h, s_w, self.c_dim], k_h=4, k_w=4, name='g_h16_deconv'))
           
      return tf.nn.sigmoid(h16)


  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.target_font, self.batch_size,
        self.target_height, self.target_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
