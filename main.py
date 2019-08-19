import tensorflow as tf 
from model import GAN_dehaze

weights_dir = "../weights/water_16/water_16.ckpt"
dataset_dir_x = '../../dehaze/data/nyu_water_rgb_no_noise.npy'
dataset_dir_y = '../../dehaze/data/nyu_gt_rgb.npy'
output_dir = "../result/output/"
imlist = ['/home/wei/code/dehaze/test/water/reef2.jpg',
'/home/wei/code/dehaze/test/water/Ancuti2.png',
'/home/wei/code/dehaze/test/water/ocean2.jpg',
'/home/wei/code/dehaze/test/water/reef1.jpg',
'/home/wei/code/dehaze/test/water/underwater-1.jpg',
'/home/wei/code/dehaze/test/water/fish.jpg',
'/home/wei/code/dehaze/test/water/Ancuti3.png',
'/home/wei/code/dehaze/test/water/Galdran_Im1_orig.png',
'/home/wei/code/dehaze/test/water/Eustice4.jpg',
'/home/wei/code/dehaze/test/water/reef3.jpg',
'/home/wei/code/dehaze/test/water/Ancuti1.png']
video_path = "/home/wei/code/dehaze/data/GOPR0216.MP4"
pretrain = True

with tf.Session() as sess:
    dehazemodel = GAN_dehaze(sess, weights_dir = weights_dir, dataset_dir_x = dataset_dir_x, dataset_dir_y = dataset_dir_y, output_dir = output_dir, pretrain = pretrain, imlist = imlist, video_path = video_path)

    #dehazemodel.train()
    #dehazemodel.test()
    #dehazemodel.images()
    dehazemodel.video()