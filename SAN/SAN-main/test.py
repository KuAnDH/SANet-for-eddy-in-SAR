# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:33:08 2022

@author: 徐花花
"""

from __future__ import division
import tensorflow as tf
import os
import numpy as np
import glob
import cv2
from stackedhg import mhourglass

def suppression(img):
    h = np.shape(img)[0]
    w = np.shape(img)[1]
    for i in range(h):
        for j in range(w):
            if img[i,j] >= 100:
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

input_dir = '/.../dataset/testing/'
gt_dir = '/.../data_aug/testgt/'
checkpoint_dir = '/.../checkpoint/'
result_dir = '/.../out/'

# get train IDs
test_fns = glob.glob(gt_dir + '*.png')
test_ids = [os.path.basename(test_fn)[0:-4] for test_fn in test_fns]


sess = tf.compat.v1.Session()
in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1])
gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1])
out_image = mhourglass(in_image)

saver =tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids:
    in_path = input_dir + str(test_id) +'.png'
    in_fn = os.path.basename(in_path)
    print(in_fn)
        
    gt_path = gt_dir + str(test_id) +'.png'
    gt_fn = os.path.basename(gt_path)
        
    in_jpg = cv2.imread(in_path,0)
    in_im=np.float32(in_jpg/255.0)
        
    gt_jpg=cv2.imread(gt_path,0)
    gt_im=np.float32(gt_jpg/255.0)
        
    input_= np.expand_dims(in_im, axis=0)
    input_ima = np.expand_dims(input_, axis=3)
    gt_ = np.expand_dims(gt_im, axis=0)
    gt_ima = np.expand_dims(gt_, axis=3)
        
    input_ima = np.minimum(input_ima, 1.0)
    output = sess.run(out_image, feed_dict={in_image: input_ima})
    output = np.minimum(np.maximum(output, 0), 1)

    output = output[0, :, :, :]
    input_out= input_ima[0,:,:,:]
    gt_out = gt_ima[0, :, :, :]
        
    cv2.imwrite(result_dir + 'final/' + str(test_id) + '_out.png', output*255.0)
    cv2.imwrite(result_dir + 'final/' + str(test_id) + '_input.png', input_out*255.0)
    cv2.imwrite(result_dir + 'final/' + str(test_id) + '_gt.png', gt_out*255.0)
     
    suppress = suppression(output*255.0)
    cv2.imwrite(result_dir + 'final/' + str(test_id) + '_suppressed.png', suppress)