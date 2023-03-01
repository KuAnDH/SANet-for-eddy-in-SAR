# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:53:58 2022

@author: pc
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def conv2d(inputs, filters, kernel=[3,3], strides=1, pad='SAME', r=1, act=False):
    x = slim.conv2d(inputs=inputs, num_outputs=filters, 
                    kernel_size=kernel, stride=strides, 
                    padding=pad, rate=r, activation_fn=None)
    if act:
        x = tf.nn.relu(x)
    return x


def convbn(inputs, filters, kernel=[3,3], strides=1, pad='SAME', r=1, act=False):
    x = slim.conv2d(inputs=inputs, num_outputs=filters, 
                    kernel_size=kernel, stride=strides, 
                    padding=pad, rate=r, activation_fn=None)
    x = tf.contrib.layers.batch_norm(x, 0.9, epsilon=1e-5, activation_fn = None)
    if act:
        x = tf.nn.relu(x) 
    return x


def bottleneck(inputs, numOut, downsample = 1):
    strides = downsample
    
    conv_1 = convbn(inputs, int(numOut/4), kernel=[1,1], strides=1, act=True)
    conv_2 = convbn(conv_1, int(numOut/4), kernel=[3,3], strides=strides, act=True)
    conv_3 = convbn(conv_2, int(numOut),   kernel=[1,1], strides=1, act=False )
    
    if downsample == 1 and inputs.get_shape().as_list()[3] == numOut:
        shortcut  = inputs
    else:
        shortcut = conv2d(inputs, numOut, kernel=[1,1], strides=strides)
    
    out = tf.nn.relu(shortcut + conv_3)
    return out


def upsample(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
    deconv_output = tf.concat([deconv, x2], 3)
    return deconv_output	

def convSpatialPyramid(inputs):
    num = int(np.shape(inputs)[3])
    
    x1 = convbn(inputs, num, kernel=[3,3], strides=1, r=1,  act=True)
    x2 = convbn(inputs, num, kernel=[3,3], strides=1, r=6,  act=True)
    x3 = convbn(inputs, num, kernel=[3,3], strides=1, r=12, act=True)
    x4 = convbn(inputs, num, kernel=[3,3], strides=1, r=18, act=True)
    
    y = tf.concat([x1,x2,x3,x4], 3)
    y = convbn(y, num, kernel=[1,1], strides=1, r=1, act=False)
    return y


def interSupervision(inputs):
    out = conv2d(inputs, 1, kernel=[1,1])
    return (out)


def GCblock(inputs, ratio):
    channel = int(np.shape(inputs)[3])
    
    context = conv2d(inputs, 1, kernel=[1,1], strides=1, act=True) #N, H, W, 1
    context = tf.reshape(context, [1, -1, 1]) #N, HW, 1
    context = tf.nn.softmax(context) #N, HW, 1
    context = tf.expand_dims(context, 1) #N, 1, HW, 1
    
    inputs_t = tf.transpose(inputs, [0,3,1,2]) #N C H W
    inputs_t = tf.reshape(inputs_t, [1, channel, -1]) #N, C, HW
    inputs_t = tf.expand_dims(inputs_t, 1) #N, 1, C, HW
    
    gc = tf.matmul(inputs_t, context) #N, 1, C, 1
    gc = tf.transpose(gc, [0, 1, 3, 2]) #N, 1, 1, C

    tr1 = convbn(gc, int(channel/ratio), kernel=[1,1], strides=1, act=True)  #N, 1, 1, C/r
    tr2 = conv2d(tr1, channel, kernel=[1,1], strides=1) # N, 1, 1, C
    
    out = inputs + tr2 
    return out


def getGatingSignal(g, numOut):
    conv = convbn(g, numOut, kernel=[1,1], strides=1, act=True)
    return conv


def gatingAttention(x, g, numOut):
    conv_x = conv2d(x, int(numOut/2), kernel=[2,2], strides=2, act=False) 
    conv_g = conv2d(g, int(numOut/2), kernel=[1,1], strides=1, act=False)
    up_g = tf.compat.v1.image.resize_bilinear(conv_g, tf.shape(conv_x)[1:3])
    add_x_g = conv_x + up_g
    tr1 = tf.nn.relu(add_x_g)
    conv_tr = conv2d(tr1, 1, kernel=[1,1], strides=1, act=False)
    tr2 = tf.nn.sigmoid(conv_tr)
    att = tf.compat.v1.image.resize_bilinear(tr2, tf.shape(x)[1:3])
    gated_x = x * att
    gated_x = convbn(gated_x, numOut, kernel=[1,1], strides=1, act=False)
    return gated_x

    
