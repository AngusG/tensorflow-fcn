#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import fcn32_vgg
import utils

from tensorflow.python.framework import ops

os.environ['CUDA_VISIBLE_DEVICES'] = ''

#img1 = skimage.io.imread("./test_data/tabby_cat.png")
parser = argparse.ArgumentParser()
parser.add_argument('--train_record', help="training tfrecord file", default="input_data_ciona_crop.tfrecords")
parser.add_argument('--npypath',help="path to weights",default="/scratch/gallowaa/")
parser.add_argument('--imgpath',help="path to input image",default="/scratch/gallowaa/224-ground-truthed/images/")
args = parser.parse_args()

#img1 = skimage.io.imread(args.imgpath+"2008_004499_02.png")
#img1 = plt.imread(args.imgpath+"IMG_5178_crop2_w112_ds2_13_x184_y120.jpg")

trn_images_batch, trn_segmentations_batch = input_pipeline(
                                                    args.train_record,
                                                    args.batch_size,
                                                    args.num_epochs)


with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn32_vgg.FCN32VGG(args.npypath+"vgg16.npy")
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])

    scp.misc.imsave('fcn32_downsampled.png', down_color)
    scp.misc.imsave('fcn32_upsampled.png', up_color)
