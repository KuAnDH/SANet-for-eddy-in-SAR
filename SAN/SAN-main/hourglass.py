# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:05:03 2022

@author: 徐花花
"""

import tensorflow as tf
from layers import convbn, upsample, convSpatialPyramid, getGatingSignal, gatingAttention


def hourglass(inputs):
    # inputs = 2048,2048,32
    
    conv1 = convbn(inputs, 32, kernel=[3,3], act=True)
    # conv1 = 2048,2048,32
    
    pool1 = tf.contrib.layers.max_pool2d(conv1, [2,2],[2,2], 'SAME')
    # pool1 = 1024,1024,32
    conv2 = convbn(pool1, 64, kernel=[3,3], act=True)
    # conv2 = 1024,1024,64
    
    pool2 = tf.contrib.layers.max_pool2d(conv2, [2,2],[2,2], 'SAME')
    # pool2 = 512,512,64
    conv3 = convbn(pool2, 128, kernel=[3,3], act=True)
    # conv3 = 512,512,128
    
    pool3 = tf.contrib.layers.max_pool2d(conv3, [2,2],[2,2], 'SAME')
    # pool3 = 256,256,128
    conv4 = convbn(pool3, 256, kernel=[3,3], act=True)
    # conv4 = 256,256,256
    
    spp = convSpatialPyramid(conv4)
    # spp = 256,256,256
    signal = getGatingSignal(spp, 256)
    # signal = 256,256,256

    gated_conv3 = gatingAttention(conv3, signal, 128)
    # gated_conv3 = 512,512,128
    gated_conv2 = gatingAttention(conv2, signal, 64)
    # gated_conv2 = 1024,1024,64
    
    
    up3 = upsample(spp, gated_conv3, 128, 256)
    # up3 = 512,512,256
    _conv3 = convbn(up3, 128, kernel=[3,3], act=True)
    # _conv3 = 512,512,128
    
    up2 = upsample(_conv3, gated_conv2, 64, 128)
    # up3 = 1024,1024,128
    _conv2 = convbn(up2, 64, kernel=[3,3], act=True)
    # conv2 = 1024,1024,64
    
    up1 = upsample(_conv2, conv1, 32, 64)
    # up4 = 2048,2048,64
    _conv1 = convbn(up1, 32, kernel=[3,3], act=True)
    # conv1 = 2048,2048,32
    
    return _conv1
