from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import cv2
import matplotlib.pyplot as plt
from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

input_height = 256
input_width = 512

params = monodepth_parameters(
        encoder='vgg',
        height=256,
        width=512,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

left  = tf.placeholder(tf.float32, [2, input_height, input_width, 3])
model = MonodepthModel(params, "test", left, None)
# SESSION
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

# SAVER
train_saver = tf.train.Saver()

# INIT
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
coordinator = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
# RESTORE
restore_path = './model/model_kitti'
train_saver.restore(sess, restore_path)
def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    #plt.imshow(l_disp,cmap='plasma')
    #plt.show()
    m_disp = 0.5 * (l_disp + r_disp)
    #plt.imshow(m_disp,cmap='plasma')
    #plt.show()
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

#return m_disp

def get_Depth_Image(image):
    input_image = image
    original_height, original_width, num_channels = input_image.shape
    input_image = cv2.resize(input_image, (input_width, input_height))
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)
    
    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
    disp_to_img = cv2.resize(disp_pp.squeeze(), (original_width, original_height))
    plt.imsave('depth_image.jpeg',disp_to_img,cmap='plasma')
    disp_to_img = cv2.imread('./depth_image.jpeg')
    return disp_to_img

