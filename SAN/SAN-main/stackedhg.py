# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:14:17 2022

@author: 徐花花
"""

import tensorflow as tf
from layers import conv2d, bottleneck, interSupervision, GCblock
from hourglass import hourglass

def mhourglass(inputs):
    #--------------preprocessing----------
    conv1 = bottleneck(inputs, 32, 1)
    #----------stacked hourglass + GCblock + intermediate supervision------
    hg = [None] * 2
    GChg = [None] * 2
    out =  [None] * 2
    final = [None] * 2
    
    
    hg[0] = hourglass(conv1)
    GChg[0] = GCblock(hg[0], ratio = 4)
    final[0] = GChg[0] + conv1
    out[0] = interSupervision(GChg[0])
    
    
    hg[1] = hourglass(final[0])
    GChg[1] = GCblock(hg[1], ratio = 4)
    final[1] = GChg[1] + final[0]
    out[1] = conv2d(final[1], 1, kernel=[1,1])
    
    stack = tf.reduce_sum(out, 0, name='outfinal')
    return stack
