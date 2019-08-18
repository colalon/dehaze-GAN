import tensorflow as tf 
from model import GAN_dehaze

weights_dir = "../weights/water_16/water_16.ckpt"
dataset_dir_x = '../../dehaze/data/nyu_water_rgb_no_noise.npy'
dataset_dir_y = '../../dehaze/data/nyu_gt_rgb.npy'
output_dir = "../result/output/"
pretrain = True

with tf.Session() as sess:
    dehazemodel = GAN_dehaze(sess, weights_dir = weights_dir, dataset_dir_x = dataset_dir_x, dataset_dir_y = dataset_dir_y, output_dir = output_dir, pretrain = pretrain)

    dehazemodel.train()