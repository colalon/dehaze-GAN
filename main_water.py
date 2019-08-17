import tensorflow as tf
from function_hazy import preprocess,deprocess,FCN,Unet_aod,Unet_atm
from crop_image import random_crop_filp,white_balance,noise_resize,color_blance_var
from crop_image import gradient_lap
import numpy as np
from scipy.misc import imsave
import time
import math
import os
import cv2

X_numpy = np.load('../../dehaze/data/nyu_water_rgb_no_noise.npy')
Y_numpy = np.load('../../dehaze/data/nyu_gt_rgb.npy')
#X_numpy = np.load('../../uw2_hazy_rgb.npy')
#Y_numpy = np.load('../../uw2_gt_rgb.npy')

X_numpy_test = X_numpy[0:]
Y_numpy_test = Y_numpy[0:]

X_raw = tf.placeholder(tf.float32, shape=(None, None,None,3))
Y_raw = tf.placeholder(tf.float32, shape=(None, None,None,3))
istrain = tf.placeholder(tf.bool, shape=[])

EPS = 1e-10
gan_weight, l1_weight = 1,100
lr, beta1 = 0.0001,0.5
switch = False
X = preprocess(X_raw)
Y = preprocess(Y_raw)

X,Y = random_crop_filp(X,Y,switch)

X_hsv = tf.image.rgb_to_hsv(X)
Y_hsv = tf.image.rgb_to_hsv(Y)

rrX = tf.concat([1 - X[:,:,:,0:1] , X[:,:,:,1:3]],axis = 3)
rrX_hsv = tf.image.rgb_to_hsv(rrX)
red = 1-tf.reduce_min(rrX,axis=3,keepdims=True)

Xr = deprocess(X)
Yr = deprocess(Y)

with tf.variable_scope("generator"):
    xk = Unet_aod(X,3)
    xk = xk + 0.5

with tf.variable_scope("generator_atm"):
    xt,xa  = Unet_atm(X,1)
    xt = 1 - (xt + 1) / 2
    xa = (xa + 1) / 2
    xtt = xt
    xt = tf.maximum(xt,0.01)

with tf.variable_scope("function"):
    jaod = (X-1) * xk + 1
    jf = (X - xa)/xt + xa
    jaod = tf.clip_by_value(jaod,0,1)
    jf = tf.clip_by_value(jf,0,1)
    jaodi = jaod*xt + xa * (1-xt)
    jfi = (jf - 1) / xk + 1
    jaodi = tf.clip_by_value(jaodi,0,1)
    jfi = tf.clip_by_value(jfi,0,1)
    xcom = (xtt) * jaod + (1.0 - xtt) * jf
    xcomw = white_balance(xcom,0.3)
    xcomb = color_blance_var(xcom)

xar = deprocess(xa)
xtr = deprocess(xt)
xkr = deprocess(1/(xk+0.5))
xcomr = deprocess(xcom)
xcombr = deprocess(xcomb)
xcomwr = deprocess(xcomw)
jaodr = deprocess(jaod)
jfr = deprocess(jf)
jaodir = deprocess(jaodi)
jfir = deprocess(jfi)
xttr = deprocess(xtt)
redr = deprocess(red)

with tf.name_scope("real_discriminator"):
    with tf.variable_scope("discriminator"):
        predict_real = FCN(Y)

with tf.name_scope("fake_discriminator"):
    with tf.variable_scope("discriminator", reuse=True):
        predict_fake = FCN(jf)
        predict_fake2 = FCN(jaod)
        hat_rand = tf.random_uniform([],0.0,1.0)
        hat = xcom * hat_rand + Y * (1.0-hat_rand)
        d_hat = FCN(hat)

ddx = tf.gradients(d_hat, hat)[0]
ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 10)


with tf.name_scope("discriminator_loss"):
    discrim_loss = tf.reduce_mean( predict_real) - tf.reduce_mean(predict_fake) + tf.reduce_mean( predict_real )- tf.reduce_mean(predict_fake2 ) + ddx
with tf.name_scope("generator_loss"):
    gen_loss_GAN = tf.reduce_mean(predict_fake) + tf.reduce_mean(predict_fake2)
    gen_loss_L1 = tf.reduce_mean(tf.abs(X - jaodi)) + tf.reduce_mean(tf.abs(X - jfi))
    gen_loss_gradient = tf.reduce_mean(tf.abs(gradient_lap(xtt) - gradient_lap(X_hsv[:,:,:,2:3])))
    gen_loss_trans = tf.reduce_mean(tf.abs(xtt - red ))
    gen_loss_imq = tf.reduce_mean(tf.abs(jaod - X )) + tf.reduce_mean(tf.abs(jf - X ))
    gen_loss = gen_loss_GAN * 10.0  +  gen_loss_L1 * 100.0 + gen_loss_gradient * 10.0 + gen_loss_trans * 10.0 + gen_loss_imq * 10.0

    
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
        
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

for k in discrim_tvars:
    print (k)

real_mean = tf.reduce_mean(predict_real[0,:,:,:])
fake_mean = tf.reduce_mean(predict_fake[0,:,:,:])

ema = tf.train.ExponentialMovingAverage(decay=0.99)
update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1,gen_loss_gradient,gen_loss_trans,gen_loss_imq])

global_step = tf.train.get_or_create_global_step()
incr_global_step = tf.assign(global_step, global_step+1)

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config=tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
sess.run( tf.global_variables_initializer() )
mode = "train"
if mode == 'train':
    saver.restore(sess, "./water_16/water_16.ckpt")
    batchsize = 4
    itersize = X_numpy.shape[0] // batchsize
    for step in range (10000000000):
        #batchx = np.random.randint(X_numpy.shape[0],size=batchsize)
        #batchy = batchx.copy()
        batchy = np.random.randint(Y_numpy.shape[0],size=batchsize)
        startx = (step % itersize)*batchsize
        X_input = X_numpy[startx:startx+batchsize]
        Y_input = Y_numpy[batchy]
        if step % 20 == 0:
            Y_input = Y_numpy[startx:startx+batchsize]
        sess.run([update_losses,discrim_train], feed_dict={X_raw: X_input, Y_raw: Y_input,istrain:True})
        if step % 10 == 0:
            sess.run([update_losses,gen_train], feed_dict={X_raw: X_input, Y_raw: Y_input,istrain:True})
        if step % 100 == 0:
            loss=sess.run([discrim_loss,gen_loss_L1,gen_loss_GAN,gen_loss_gradient,gen_loss_trans,gen_loss_imq], feed_dict={X_raw: X_input, Y_raw: Y_input,istrain:False})
            print (loss)
            xin,yin,imt,imcom,ima,imjaod,imjf,imjaodi,imjfi,imcomb,imtt,imcomw,imxk,imred = sess.run([Xr,Yr,xtr,xcomr,xar,jaodr,jfr,jaodir,jfir,xcombr,xttr,xcomwr,xkr,redr], feed_dict={X_raw: X_input, Y_raw: Y_input,istrain:False})
            print (ima[0])
            imsave('./output/in.jpg',np.uint8(X_numpy[startx]))
            imsave('./output/out.jpg',np.uint8(Y_numpy[startx]))
            imsave('./output/x.jpg',np.uint8(xin[0]))
            imsave('./output/y.jpg',np.uint8(yin[0]))
            imsave('./output/imt.jpg',np.uint8(imt[0,:,:,0]))
            imsave('./output/imcom.jpg',np.uint8(imcom[0]))
            imsave('./output/imjaod.jpg',np.uint8(imjaod[0]))
            imsave('./output/imjf.jpg',np.uint8(imjf[0]))
            imsave('./output/imjaodi.jpg',np.uint8(imjaodi[0]))
            imsave('./output/imjfi.jpg',np.uint8(imjfi[0]))
            imsave('./output/imcomb.jpg',np.uint8(imcomb[0]))
            imsave('./output/imcomw.jpg',np.uint8(imcomw[0]))
            imsave('./output/imxk.jpg',np.uint8(imxk[0]))
            imsave('./output/imred.jpg',np.uint8(imred[0,:,:,0]))
            imsave('./output/imtt.jpg',np.uint8(imtt[0,:,:,0]))
            print (step)
            saver.save(sess, "./water_16/water_16.ckpt")
            if step % itersize == itersize - 1:
                rand_ = np.random.permutation(X_numpy.shape[0])
                X_numpy = X_numpy[rand_]
                Y_numpy = Y_numpy[rand_]

elif mode == "test":
    saver.restore(sess, "../weights/water_16/water_16.ckpt")
    for i in range (X_numpy_test.shape[0]):
        im = X_numpy_test[i]
        #im = cv2.resize(im,(512,512))
        im = np.expand_dims(im,axis=0)
        imcom,imjf,imjaod,imt,imaodi,imfi,imk=sess.run([xcomr,jfr,jaodr,xtr,jaodir,jfir,xkr], feed_dict={X_raw:im, istrain:False})
        target = Y_numpy_test[i].copy()
        #out = np.hstack([im[0],imjc[0],target])
        imsave("../result/output_all/"+str(i)+"truth.png",target)
        imsave("../result/output_all/"+str(i)+"result.png",np.uint8(imcom[0]))
        imsave("../result/output_all/"+str(i)+"trans.png",imt[0,:,:,0])
        imsave("../result/output_all/"+str(i)+"input.png",np.uint8(im[0]))
        imsave("../result/output_all/"+str(i)+"jf.png",np.uint8(imjf[0]))
        imsave("../result/output_all/"+str(i)+"jaod.png",np.uint8(imjaod[0]))
        imsave("../result/output_all/"+str(i)+"aodi.png",np.uint8(imaodi[0]))
        imsave("../result/output_all/"+str(i)+"fi.png",np.uint8(imfi[0]))
        imsave("../result/output_all/"+str(i)+"imk.png",np.uint8(imk[0]))
        print (i)
elif mode == "one_pic":
    saver.restore(sess, "./hazy_16/hazy_16.ckpt")
    #for i in range (X_numpy_test.shape[0]):
    im = cv2.imread("../hazy/e.jpg")
    im = cv2.resize(im,(512,512))
    im = np.expand_dims(im,axis=0)
    imjc=sess.run(xjr, feed_dict={X_raw:im, istrain:False})
    imjc = np.clip(imjc, 0, 255)
    cv2.imwrite("./output/"+"result.png",imjc[0])
elif mode == "pic":
    saver.restore(sess, "./na_water_16/na_water_16.ckpt")
    with open("/home/dipxyz/code/dehaze/test/water.txt") as f:
        content = f.read().splitlines()
    print (content)
    for i in range (len(content)):
        im = cv2.imread(content[i])
        #print (im)
        im = cv2.resize(im,(256,256))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = np.expand_dims(im,axis=0)
        imjc,imjc2,imjc3=sess.run([jfr,xttr,xcomwr], feed_dict={X_raw:im, istrain:False})
        imjc = np.clip(imjc, 0, 255)
        imjc2 = np.clip(imjc2, 0, 255)
        imjc3 = np.clip(imjc3, 0, 255)
        imsave("./output/"+"input"+str(i)+".png",np.uint8(im[0]))
        imsave("./output/"+"result"+str(i)+".png",np.uint8(imjc[0]))
        imsave("./output/"+"trans"+str(i)+".png",np.uint8(imjc2[0,:,:,0]))
        imsave("./output/"+"rw"+str(i)+".png",np.uint8(imjc3[0]))
else:
    saver.restore(sess, "./na_water_16/na_water_16.ckpt")
    cap = cv2.VideoCapture('../data/GP010215.MP4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('noutput_cat.avi', fourcc, fps, (512,512))
    while 1:
        ret , frame = cap.read()
        print (ret)
        if ret == True:
            frame = cv2.resize(frame,(512,256))
            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            judge = np.expand_dims(frame_rgb,axis=0)
            #judge = batch_he_cpu(judge)
            imjc,imt,aa = sess.run([xcomr,xtr,xar], feed_dict={X_raw:judge,istrain:False})
            print (aa)
            imjc = np.clip(imjc, 0, 255)
            imjc = np.uint8(imjc)
            #imjc = batch_he_cpu(imjc)
            imt = np.uint8(imt)
            imjc = cv2.cvtColor(imjc[0],cv2.COLOR_RGB2BGR)
            imt = cv2.cvtColor(imt[0],cv2.COLOR_RGB2BGR)
            cv2.imshow('hazy',frame)
            cv2.imshow('result',imjc)
            cv2.imshow('imt',imt)
            out.write(np.vstack((frame,imjc)))
            cv2.waitKey(1)
