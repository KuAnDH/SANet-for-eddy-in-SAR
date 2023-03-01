# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:19:08 2022

@author: 徐花花
"""

from __future__ import division
import os, time
import tensorflow as tf
import numpy as np
import glob
import cv2
from stackedhg import mhourglass
from losses import mixedLoss

input_dir = '/.../dataset/training/'
gt_dir = '/.../dataset/traingt/'
checkpoint_dir = '/.../checkpoint/'
result_dir = '/.../out/'


# get train IDs
train_fns = glob.glob(gt_dir +  '*.png')
train_ids = [os.path.basename(train_fn)[0:-4] for train_fn in train_fns]

save_freq = 10

sess = tf.compat.v1.Session()
in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1])
gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1])
out_image = mhourglass(in_image)

G_loss = mixedLoss(out_image, gt_image)

t_vars = tf.compat.v1.trainable_variables()#返回当前图中trainable=true的变量
lr = tf.compat.v1.placeholder(tf.float32)#learning rate 可变
G_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(G_loss) #优化器Adam optimizer

saver = tf.compat.v1.train.Saver()#保存模型
sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)#加载训练好的模型
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir+'0*')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-5
for epoch in range(lastepoch, 400):
    if os.path.isdir(result_dir+"%04d" % epoch):#判断是否是目录,若是则从下一个epoch开始
        continue                               #若不是，则epoch不间断
    cnt = 0
    if epoch > 200:
        learning_rate = 1e-6
    if epoch > 345:
        learning_rate = 1e-7

    for i in range(len(train_ids)):
        # get the path from image id
        train_id = train_ids[i]
        in_path = input_dir + str(train_id) + '.png'
        in_fn = os.path.basename(in_path)
        print(in_fn)
        
        in_jpg = cv2.imread(in_path,0)
        in_im=np.float32(in_jpg/255.0)
        

        gt_path = gt_dir + str(train_id) + '.png'
        gt_fn = os.path.basename(gt_path)
        
        gt_png=cv2.imread(gt_path,0)
        gt_im=np.float32(gt_png/255.0)
        
        input_ = np.expand_dims(in_im, axis=0)
        input_ima = np.expand_dims(input_, axis=3)
        gt_ = np.expand_dims(gt_im, axis=0)
        gt_ima = np.expand_dims(gt_, axis=3)
    

        st = time.time()
        cnt += 1
        
        input_ima = np.minimum(input_ima, 1.0)
        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: input_ima, gt_image: gt_ima, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[i] = G_current

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)
                nowdir = result_dir + '%04d/' % epoch
            
            temp = output[0, :, :, :]
            cv2.imwrite(nowdir + str(train_id) + '.png', temp*255.0)
       
    
    saver.save(sess, checkpoint_dir + 'model.ckpt')
    if epoch % save_freq == 0:
        saver.save(sess, nowdir + 'model.ckpt')