# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:07:28 2022

@author: 徐花花
"""
import tensorflow as tf

def mae(output, target):
    mae = tf.reduce_mean(tf.square(output - target)) #L2loss/MSE
    return mae

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def focal_loss(output, target, alpha=0.25, gamma=2): 
        zeros = tf.zeros_like(output)
        pos_p_sub = tf.where(target > zeros, target - output, zeros) # positive sample 寻找正样本，并进行填充
        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(target > zeros, zeros, output) # negative sample 寻找负样本，并进行填充
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(output, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - output, 1e-8, 1.0))

        return tf.reduce_sum(per_entry_cross_ent)

def mixedLoss(output, target, alpha=1e-5,):
    return alpha * focal_loss(output,target) - tf.math.log(tf.abs(dice_coe(output,target)))