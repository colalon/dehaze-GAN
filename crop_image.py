import tensorflow as tf
import numpy as np

def random_crop_filp(im_hazy_in,im_gt_in,switch=False):

    if switch == False:
        hazy = im_hazy_in
        gt = im_gt_in
    else:
        re_size = 286
        size = 256
        im_hazy = tf.image.resize_images(im_hazy_in,[re_size,re_size])
        im_gt = tf.image.resize_images(im_gt_in,[re_size,re_size])
        im_hazy = tf.image.random_flip_left_right(im_hazy)
        im_gt = tf.image.random_flip_left_right(im_gt)
        im_hazy = tf.image.random_flip_up_down(im_hazy)
        im_gt = tf.image.random_flip_up_down(im_gt)
        #crop_size = tf.random_uniform([],2,8)*32
        crop_size = size#tf.cast(crop_size, dtype=tf.int32)
        #print (crop_size)
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, re_size - crop_size + 1)), dtype=tf.int32)
        #print (offset)
        hazy = tf.image.crop_to_bounding_box(im_hazy, offset[0], offset[1], crop_size, crop_size)
        gt = tf.image.crop_to_bounding_box(im_gt, offset[0], offset[1], crop_size, crop_size)

    return hazy,gt


def white_balance(im,alpha):
    eps = 1e-3
    gray = 0.299 * im[:,:,:,0:1] + 0.587 * im[:,:,:,1:2] + 0.114 * im[:,:,:,2:3]
    Lmean = tf.reduce_mean(gray,axis=[1,2,3],keepdims=True)
    imp = tf.pow( (gray / (im+eps) ),alpha ) * im
    pmean = tf.reduce_mean(imp,axis=[1,2],keepdims=True)
    result = (imp - pmean) + Lmean
    result = tf.clip_by_value(result,0,1)
    return result

def color_blance_var(im):
    mean,var = tf.nn.moments(im,[1,2,3],keep_dims = True)
    std = tf.sqrt(var)
    imr = (im - mean) / std
    im_min = -2
    im_max = 2
    imr = (imr - im_min) / (im_max-im_min)
    imr = tf.clip_by_value(imr,0,1)
    return imr

def noise_resize(im,switch=True):
    if switch == False:
        return im
    else:
        offset = tf.cast(tf.floor(tf.random_uniform([2], 128, 200)), dtype=tf.int32)
        temp = tf.image.resize_bilinear(im,[offset[0],offset[1]])
        temp = tf.image.resize_bilinear(temp,[256,256])
        noise_rate = tf.random_uniform([],0.01,0.05)
        rand = tf.random_normal([1,256,256,3],stddev=noise_rate)
        result =  temp + rand
        result = tf.clip_by_value(result,0.0,1.0)
        #beta = tf.random_uniform([],0.6,1.4)
        #result = tf.pow(result,beta)
        isnoise = tf.random_uniform([],0,1)
        result = tf.cond(isnoise < 0.5, lambda: im, lambda: result)

        return result

def gradient_lap(im):
    trans_weight = tf.constant([1.0,1.0,1.0,1.0,-8.0,1.0,1.0,1.0,1.0])
    trans_w_v = tf.reshape(trans_weight,[3,3,1,1])
    trans_v = tf.nn.conv2d(im,trans_w_v,[1,1,1,1],"SAME")
    trans_loss = tf.abs(trans_v)

    return trans_loss

