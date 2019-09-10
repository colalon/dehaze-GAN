import tensorflow as tf
import numpy as np
from scipy.misc import imsave,imread
import cv2
from ops import preprocess,deprocess,FCN,Unet_aod,Unet_atm
from utils import gradient_lap,random_crop_filp,white_balance

class GAN_dehaze():
    def __init__(self, sess, batch_size=4, ngf=16, ndf=16, EPS = 1e-10, gan_weight = 1., l1_weight = 100., lr = 0.0002, beta1 = 0.5, weights_path = None, dataset_path_x = None, dataset_path_y = None, output_dir = None, pretrain = True, im_list = None, video_path = None):
        self.sess = sess
        self.batch_size = batch_size
        self.ngf = ngf
        self.ndf = ndf
        self.EPS = EPS
        self.gan_weight = gan_weight
        self.l1_weight = l1_weight
        self.lr = lr
        self.beta1 = beta1
        self.weights_path = weights_path
        self.dataset_path_x = dataset_path_x
        self.dataset_path_y = dataset_path_y
        self.output_dir = output_dir
        self.pretrain = pretrain
        self.im_list = im_list
        self.video_path = video_path


        self.build_model()

    def build_model(self):

        self.X_raw = tf.placeholder(tf.float32, shape=(None, None,None,3))
        self.Y_raw = tf.placeholder(tf.float32, shape=(None, None,None,3))
        self.istrain = tf.placeholder(tf.bool, shape=[])

        self.X = preprocess(self.X_raw)
        self.Y = preprocess(self.Y_raw)

        self.Xf,self.Yf = random_crop_filp(self.X,self.Y,True)

        self.X = tf.cond(self.istrain,lambda: self.Xf, lambda: self.X)
        self.Y = tf.cond(self.istrain,lambda: self.Yf, lambda: self.Y)

        self.X_hsv = tf.image.rgb_to_hsv(self.X)
        self.Y_hsv = tf.image.rgb_to_hsv(self.Y)

        self.rrX = tf.concat([1 - self.X[:,:,:,0:1] , self.X[:,:,:,1:3]],axis = 3)
        self.red = 1-tf.reduce_min(self.rrX,axis=3,keepdims=True)

        with tf.variable_scope("generator"):
            self.xk = Unet_aod(self.X,3)
            self.xk = self.xk + 0.5

        with tf.variable_scope("generator_atm"):
            self.xt,self.xa  = Unet_atm(self.X,1)
            self.xt = 1 - (self.xt + 1) / 2
            self.xa = (self.xa + 1) / 2
            self.xtt = self.xt
            self.xt = tf.maximum(self.xt,0.01)

        with tf.variable_scope("function"):
            self.jaod = (self.X-1) * self.xk + 1
            self.jf = (self.X - self.xa) / self.xt + self.xa
            self.jaod = tf.clip_by_value(self.jaod,0,1)
            self.jf = tf.clip_by_value(self.jf,0,1)
            self.jaodi = self.jaod * self.xt + self.xa * (1-self.xt)
            self.jfi = (self.jf - 1) / self.xk + 1
            self.jaodi = tf.clip_by_value(self.jaodi,0,1)
            self.jfi = tf.clip_by_value(self.jfi,0,1)
            self.xcom = (self.xtt) * self.jaod + (1.0 - self.xtt) * self.jf
            self.xcomw = white_balance(self.xcom,0.2)

        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                self.predict_real = FCN(self.Y)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                self.predict_fake = FCN(self.jf)
                self.predict_fake2 = FCN(self.jaod)
                hat_rand = tf.random_uniform([],0.0,1.0)
                self.hat = self.xcom * hat_rand + self.Y * (1.0 - hat_rand)
                self.d_hat = FCN(self.hat)

        self.ddx = tf.gradients(self.d_hat, self.hat)[0]
        self.ddx = tf.sqrt(tf.reduce_sum(tf.square(self.ddx), axis=1))
        self.ddx = tf.reduce_mean(tf.square(self.ddx - 1.0) * 10)


        with tf.name_scope("discriminator_loss"):
            self.discrim_loss = tf.reduce_mean( self.predict_real) - tf.reduce_mean( self.predict_fake) + tf.reduce_mean( self.predict_real )- tf.reduce_mean( self.predict_fake2 ) + self.ddx
        with tf.name_scope("generator_loss"):
            self.gen_loss_GAN = tf.reduce_mean(self.predict_fake) + tf.reduce_mean(self.predict_fake2)
            self.gen_loss_L1 = tf.reduce_mean(tf.abs(self.X - self.jaodi)) + tf.reduce_mean(tf.abs(self.X - self.jfi))
            self.gen_loss_gradient = tf.reduce_mean(tf.abs(gradient_lap(self.xtt) - gradient_lap(self.X_hsv[:,:,:,2:3])))
            self.gen_loss_trans = tf.reduce_mean(tf.abs(self.xtt - self.red ))
            self.gen_loss_imq = tf.reduce_mean(tf.abs(self.jaod - self.X )) + tf.reduce_mean(tf.abs(self.jf - self.X ))
            self.gen_loss = self.gen_loss_GAN * 10.0  +  self.gen_loss_L1 * 100.0 + self.gen_loss_gradient * 10.0 + self.gen_loss_trans * 10.0 + self.gen_loss_imq * 10.0

    
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope("discriminator_train"):
                self.discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
                discrim_optim = tf.train.AdamOptimizer(self.lr, self.beta1)
                discrim_grads_and_vars = discrim_optim.compute_gradients(self.discrim_loss, var_list=self.discrim_tvars)
                self.discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
        
        with tf.name_scope("generator_train"):
            with tf.control_dependencies([self.discrim_train]):
                self.gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.lr, self.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(self.gen_loss, var_list=self.gen_tvars)
                self.gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        #for k in discrim_tvars:
        #    print (k)

        self.real_mean = tf.reduce_mean(self.predict_real[0,:,:,:])
        self.fake_mean = tf.reduce_mean(self.predict_fake[0,:,:,:])

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_losses = ema.apply([self.discrim_loss, self.gen_loss_GAN, self.gen_loss_L1, self.gen_loss_gradient, self.gen_loss_trans, self.gen_loss_imq])

        self.global_step = tf.train.get_or_create_global_step()
        self.incr_global_step = tf.assign(self.global_step, self.global_step+1)
        self.saver = tf.train.Saver()

    def train(self):
        X_numpy = np.load(self.dataset_path_x)
        Y_numpy = np.load(self.dataset_path_y)
        
        self.sess.run(tf.global_variables_initializer())

        if self.pretrain == True:
            self.saver.restore(self.sess, self.weights_path)
        
        for step in range (10000000000):
            batchx = np.random.randint(X_numpy.shape[0],size=self.batch_size)
            batchy = np.random.randint(Y_numpy.shape[0],size=self.batch_size)
            X_input = X_numpy[batchx]
            Y_input = Y_numpy[batchy]

            if step % 20 == 0:
                Y_input = Y_numpy[batchx]
            self.sess.run([self.update_losses,self.discrim_train], feed_dict={self.X_raw: X_input, self.Y_raw: Y_input,self.istrain:True})

            #if step % 10 == 0:
            #    self.sess.run([self.update_losses,self.gen_train], feed_dict={self.X_raw: X_input, self.Y_raw: Y_input,self.istrain:True})
            if step % 100 == 0:
                loss=self.sess.run([self.discrim_loss,self.gen_loss_L1,self.gen_loss_GAN,self.gen_loss_gradient,self.gen_loss_trans,self.gen_loss_imq], feed_dict={self.X_raw: X_input, self.Y_raw: Y_input, self.istrain:False})
                print (loss)
                #xin,yin,imt,imcom,ima,imjaod,imjf,imjaodi,imjfi,imcomb,imtt,imcomw,imxk,imred = self.sess.run([Xr,Yr,xtr,xcomr,xar,jaodr,jfr,jaodir,jfir,xcombr,xttr,xcomwr,xkr,redr], feed_dict={self.X_raw: X_input, self.Y_raw: Y_input,self.istrain:False})
                xin,ima,imcom = self.sess.run([self.X,self.xa,self.xcom], feed_dict={self.X_raw: X_input, self.Y_raw: Y_input,self.istrain:False})
                print (ima[0]*255.0)
                imsave(self.output_dir+'xin.jpg',np.uint8(X_input[0]))
                imsave(self.output_dir+'yin.jpg',np.uint8(Y_input[0]))
                imsave(self.output_dir+'x.jpg',np.uint8(xin[0]*255.0))
                imsave(self.output_dir+'imcom.jpg',np.uint8(imcom[0]*255.0))
                print (step)
                self.saver.save(self.sess, "./water_16/water_16.ckpt")
    
    def test(self):
        X_numpy = np.load(self.dataset_path_x)
        Y_numpy = np.load(self.dataset_path_y)
        
        self.sess.run(tf.global_variables_initializer())

        if self.pretrain == True:
            self.saver.restore(self.sess, self.weights_path)
        
        for i in range (X_numpy.shape[0]):
            im = X_numpy[i:i+1]
            imcom,imcomw,imjf,imjaod,imt,imaodi,imfi,imk = self.sess.run([self.xcom,self.xcomw,self.jf,self.jaod,self.xt,self.jaodi,self.jfi,self.xk], feed_dict={self.X_raw:im, self.istrain:False})
            target = Y_numpy[i].copy()
            #out = np.hstack([im[0],imjc[0],target])
            imsave(self.output_dir+str(i)+"truth.png",target)
            imsave(self.output_dir+str(i)+"result.png",np.uint8(imcom[0]*255.0))
            imsave(self.output_dir+str(i)+"resultw.png",np.uint8(imcomw[0]*255.0))
            imsave(self.output_dir+str(i)+"trans.png",imt[0,:,:,0]*255.0)
            imsave(self.output_dir+str(i)+"input.png",np.uint8(im[0]*255.0))
            imsave(self.output_dir+str(i)+"jf.png",np.uint8(imjf[0]*255.0))
            imsave(self.output_dir+str(i)+"jaod.png",np.uint8(imjaod[0]*255.0))
            imsave(self.output_dir+str(i)+"aodi.png",np.uint8(imaodi[0]*255.0))
            imsave(self.output_dir+str(i)+"fi.png",np.uint8(imfi[0]*255.0))
            imsave(self.output_dir+str(i)+"imk.png",np.uint8((1/(imk[0]+0.5))*255.0))
            print (i)
    
    def images(self):

        self.sess.run(tf.global_variables_initializer())

        if self.pretrain == True:
            self.saver.restore(self.sess, self.weights_path)

        for i,filename in enumerate(self.im_list):
            im = imread(self.im_list[i])
            im = cv2.resize(im[:,:,0:3],(256,256))
            im = np.expand_dims(im,axis=0)
            imcom, imcomw, imt = self.sess.run([self.xcom, self.xcomw, self.xtt], feed_dict={self.X_raw:im, self.istrain:False})
            imsave(self.output_dir+"input"+str(i)+".png",np.uint8(im[0]))
            imsave(self.output_dir+"result"+str(i)+".png",np.uint8(imcom[0]*255.0))
            imsave(self.output_dir+"resultw"+str(i)+".png",np.uint8(imcomw[0]*255.0))
            imsave(self.output_dir+"trans"+str(i)+".png",np.uint8(imt[0,:,:,0]*255.0))
            print (i,filename)
    
    def video(self):

        self.sess.run(tf.global_variables_initializer())

        if self.pretrain == True:
            self.saver.restore(self.sess, self.weights_path)
            print ("load model")

        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(self.output_dir+'output.avi', fourcc, fps, (512,256))
        while 1:
            ret , frame = cap.read()
            print (ret)
            if ret == True:
                frame = cv2.resize(frame,(256,256))
                frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                judge = np.expand_dims(frame_rgb,axis=0)
                #judge = batch_he_cpu(judge)
                imjc,imt,ima = self.sess.run([self.xcomw,self.xt,self.xa], feed_dict={self.X_raw:judge,self.istrain:False})
                print (ima*255.0)
                imjc = np.uint8(imjc*255.0)
                #imjc = batch_he_cpu(imjc)
                imt = np.uint8(imt*255.0)
                imjc = cv2.cvtColor(imjc[0],cv2.COLOR_RGB2BGR)
                imt = cv2.cvtColor(imt[0],cv2.COLOR_RGB2BGR)
                cv2.imshow('hazy',frame)
                cv2.imshow('result',imjc)
                cv2.imshow('imt',imt)
                out.write(np.hstack((frame,imjc)))
                cv2.waitKey(1)


    