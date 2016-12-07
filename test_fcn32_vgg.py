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
#img1 = skimage.io.imread(args.imgpath+"2008_004499_02.png")
#img1 = plt.imread(args.imgpath+"IMG_5178_crop2_w112_ds2_13_x184_y120.jpg")

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            #'mask_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    # must be read back as uint8 here
    #label = tf.cast(features['label'], tf.int32)
    
    h = tf.cast(features['height'], tf.int32)
    w = tf.cast(features['width'], tf.int32)
    d = tf.cast(features['depth'], tf.int32)
    '''
    h = features['height']
    w = features['width']
    d = features['depth']
    '''
    i = tf.decode_raw(features['image_raw'], tf.uint8)
    
    #segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)

    i.set_shape([224*224*3])
    #i.set_shape([h*w*d])
    #segmentation.set_shape([224*224*1])

    #image = tf.reshape(image,[224,224,3])
    i = tf.reshape(i,[h,w,d])
    #segmentation = tf.reshape(segmentation,[1,224,224])
    
    #return image, segmentation
    return h,w,d,i

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        [filenames], num_epochs=num_epochs,shuffle=False)

    #image, label = read_and_decode(filename_queue)
    height,width,depth,image=read_and_decode(filename_queue)

    '''
    images_batch, labels_batch = tf.train.batch(
        [image, label], 
        enqueue_many=True,
        batch_size=batch_size,
        allow_smaller_final_batch=True,
        )
    return images_batch, labels_batch
    '''
    '''
    images_batch = tf.train.batch(
        [image], 
        enqueue_many=True,
        batch_size=batch_size,
        allow_smaller_final_batch=True,
        )
    '''
    image_batch = tf.expand_dims(image, 0)    
    return image_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec', help="training tfrecord file", default="/scratch/gallowaa/pascalvoc2012.tfrecords")
    parser.add_argument('--npypath',help="path to weights",default="/scratch/gallowaa/")
    parser.add_argument('--ep', help="number of epochs.", type=int, default=50)
    parser.add_argument('--bs', help="batch size", type=int, default=1)
    parser.add_argument('--savepath',help="path to input image", \
    default="/scratch/gallowaa/224-ground-truthed/fcn32-pred/")
    #parser.add_argument('--imgpath',help="path to input image",default="/scratch/gallowaa/224-ground-truthed/images/")
    args = parser.parse_args()

    print args.npypath+args.rec


    with tf.Graph().as_default():

        #batch_images, batch_segmentations = input_pipeline(args.npypath+args.rec, args.bs)
        single_image = input_pipeline(args.npypath+args.rec, args.bs)

        init = tf.initialize_all_variables()
        #init_locals = tf.initialize_local_variables()

        #image_holder=tf.placeholder("float")
    
        with tf.Session() as sess:

            vgg_fcn = fcn32_vgg.FCN32VGG(args.npypath+"vgg16.npy")

            #sess.run([init, init_locals])
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            #with tf.name_scope("content_vgg"):
            vgg_fcn.build(single_image, debug=True)

            print('Finished building Network.')     
            print('Running the Network')

            for i in range(1):
                
                try:
                    img = sess.run([single_image])   
                    #feed_dict = {image_holder: single_image}
                    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
                    #down, up = sess.run(tensors, feed_dict=feed_dict)                   
                    down, up = sess.run(tensors)                   

                    #down_color = utils.color_image(down[0])
                    #up_color = utils.color_image(up[0])

                    #scp.misc.imsave('fcn32_downsampled.png', down_color)
                    #scp.misc.imsave('fcn32_upsampled.png', up_color)
                    #scp.misc.imsave(args.savepath+'fcn32_upsampled_'+str(i)+'.png', up_color)

                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    coord.request_stop()
                    coord.join(threads)