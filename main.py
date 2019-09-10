import tensorflow as tf 
from model import GAN_dehaze
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--phase', dest='phase', default='test', help='train, test, images video')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
parser.add_argument('--ngf', dest='ngf', type=int, default=16, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=16, help='# of discri filters in first conv layer')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--weights_path', dest='weights_path', default='./weights', help='models are saved here')
parser.add_argument('--dataset_path_x', dest='dataset_path_x', default='./dataset', help='path of the dataset')
parser.add_argument('--dataset_path_y', dest='dataset_path_y', default='./dataset', help='path of the dataset')
parser.add_argument('--output_dir', dest='output_dir', default='./output/', help='sample are saved here')
parser.add_argument('--pretrain', dest='pretrain', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--im_list', dest='im_list', nargs='+', default='[sample1.jpg,sample2.png]', help='a list of images for dehazing')
parser.add_argument('--video_path', dest='video_path', default='./sample.avi', help='name of the video')

'''
weights_path = "../weights/water_16/water_16.ckpt"
dataset_path_x = '../../dehaze/data/nyu_water_rgb_no_noise.npy'
dataset_path_y = '../../dehaze/data/nyu_gt_rgb.npy'
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
'''

args = parser.parse_args()
print (args.pretrain,"fsgdfhgh")
with tf.Session() as sess:
    dehazemodel = GAN_dehaze(sess, weights_path = args.weights_path, 
        dataset_path_x = args.dataset_path_x, 
        dataset_path_y = args.dataset_path_y, 
        output_dir = args.output_dir, 
        pretrain = args.pretrain, 
        im_list = args.im_list, 
        video_path = args.video_path)

    if args.phase == "train":
        dehazemodel.train()
    if args.phase == "test":
        dehazemodel.test()
    if args.phase == "images":
        dehazemodel.images()
    if args.phase == "video":
        dehazemodel.video()