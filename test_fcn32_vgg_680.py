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

import time
from datetime import datetime

from tensorflow.python.framework import ops

os.environ['CUDA_VISIBLE_DEVICES'] = ''

#img1 = skimage.io.imread("./test_data/tabby_cat.png")
#img1 = skimage.io.imread(args.imgpath+"2008_004499_02.png")
#img1 = plt.imread(args.imgpath+"IMG_5178_crop2_w112_ds2_13_x184_y120.jpg")

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    # must be read back as uint8 here
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)

    image.set_shape([224*224*3])
    segmentation.set_shape([224*224*1])

    image = tf.reshape(image,[224,224,3])
    segmentation = tf.reshape(segmentation,[224,224])

    rgb = tf.cast(image, tf.float32)
    rgb = rgb * (1./255)
    rgb = tf.cast(image, tf.float32)

    mask = tf.cast(segmentation, tf.float32)
    mask = (mask / 255.) * 20
    #mask = tf.cast(mask, tf.int32)
    mask = tf.cast(mask, tf.int64)
    
    return rgb, mask

def input_pipeline(filenames, batch_size, num_epochs):
    filename_queue = tf.train.string_input_producer(
        [filenames], num_epochs=num_epochs,shuffle=False)

    image, label = read_and_decode(filename_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    images_batch, labels_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        enqueue_many=False, shapes=None,
        allow_smaller_final_batch=True,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return images_batch, labels_batch

def loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        logits = logits + epsilon
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits)

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec', help="training tfrecord file", default="pascalvoc2012.tfrecords")
    parser.add_argument('--npypath',help="path to weights",default="/scratch/gallowaa/")
    parser.add_argument('--log_dir', help="where to log training", default="680_log")
    parser.add_argument('--ep', help="number of epochs.", type=int, default=50)
    parser.add_argument('--lr',help="learning rate",type=float, default=10e-4)
    parser.add_argument('--bs', help="batch size", type=int, default=20)
    parser.add_argument('--savepath',help="path to input image", \
    default="/scratch/gallowaa/tensorflow-fcn-pascal")
    parser.add_argument('--num_classes', help="number of semantic classes", type=int, default=21)
    #parser.add_argument('--imgpath',help="path to input image",default="/scratch/gallowaa/224-ground-truthed/images/")
    args = parser.parse_args()

    print args.npypath+args.rec


    #with tf.Graph().as_default():

    batch_images,batch_segmentations=input_pipeline(args.npypath+args.rec,\
                                                    args.bs, \
                                                    args.ep)

    vgg_fcn = fcn32_vgg.FCN32VGG(args.npypath+"vgg16.npy")
    vgg_fcn.build(batch_images, train=True, \
        num_classes=args.num_classes, \
        random_init_fc8=True,
        debug=False)
    #up=vgg_fcn.pred_up
    up=vgg_fcn.upscore

    print 'up before reshape'
    print up.get_shape()
    logits=tf.reshape(up, (-1, args.num_classes))
    print 'up after reshape'
    print logits.get_shape()
    #logits=tf.cast(logits,tf.float32)

    print 'batch shape' 
    print batch_segmentations.get_shape()
    
    #loss = loss(logits, batch_segmentations, args.num_classes, head=None)
    
    labels = tf.reshape(batch_segmentations, [-1])
    print 'batch reshape' 
    print labels.get_shape()
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits, \
        labels, name='x_entropy')
        #tf.cast(labels, tf.int64), name='x_entropy')
    loss=tf.reduce_mean(cross_entropy, name='x_entropy_mean')

    # get variables of top layers to train during fine-tuning
    trainable_layers = ["up"]
    train_vars = []
    for idx in trainable_layers:
        train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,idx)
        
    train_step=tf.train.AdamOptimizer(args.lr).minimize(loss,var_list=train_vars)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    print('Finished building Network.')     

    init = tf.initialize_all_variables()
    init_locals = tf.initialize_local_variables()

    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config=config) as sess:

        sess.run([init, init_locals])
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        summary_writer = tf.train.SummaryWriter(args.log_dir, sess.graph)
        training_summary = tf.scalar_summary("loss", loss)

        #with tf.name_scope("content_vgg"):
        #vgg_fcn.build(single_image, debug=True)

        print('Running the Network')

        try:
            step=0
            while not coord.should_stop():

                start_time = time.time()
                _,loss_val,train_sum=sess.run([train_step,loss,training_summary])
                #_, loss_val=sess.run([train_step,loss])
                elapsed=time.time()-start_time
                summary_writer.add_summary(train_sum, step)

                assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

                if step % 1 == 0:
                    num_examples_per_step = args.bs
                    examples_per_sec = num_examples_per_step / elapsed
                    sec_per_batch = float(elapsed)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_val,
                         examples_per_sec, sec_per_batch))
                
                if step % 1 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                step+=1

        except tf.errors.OutOfRangeError:
            print 'Done training -- epoch limit reached'
        finally:
            coord.request_stop()
            coord.join(threads)
