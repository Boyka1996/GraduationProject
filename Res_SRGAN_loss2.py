"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division

import math
import cv2
from keras.layers import *
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from data_loader_s2 import DataLoader
import numpy as np
import os
from skimage.measure import compare_ssim
import tensorflow as tf
import keras.backend as K

# num_classes = 10
# batch_size = 64         # 64 or 32 or other
# epochs = 300
iterations = 782
USE_BN=True
DROPOUT=0.2 # keep 80%
CONCAT_AXIS=3
weight_decay=5e-2
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
log_filepath  = './my_resnext'

class SRGAN():
    def __init__(self):
        # 低分辨率图像的shape
        self.channels = 3
        self.lr_height = 128
        self.lr_width = 128
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)

        # 高分辨率图的shape
        self.hr_height = self.lr_height * 4
        self.hr_width = self.lr_width * 4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # 16个残差卷积块
        self.n_residual_blocks = 16  # 就是那个B
        # 优化器
        optimizer = Adam(0.0001, 0.5)

        # 采用VGG19的预训练模型来分别从高分辨率图像和生成的fake高分辨率图像中中提取图像特征，
        # 并最小化他们的MSE均方误差
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        # self.vgg.summary()
        self.vgg.compile(loss='mse',
                         optimizer=optimizer,  # 所使用的优化器，在前面默认成Adam
                         metrics=['accuracy'])  # 列表，包含评估模型在训练和测试时的性能的指标

        # 数据集
        self.dataset_name = 'DIV'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)计算判别器D的输出维度。这里采用的是patchGAN
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # 建立判别器模型
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # self.discriminator.summary()

        # 建立生成器模型
        self.generator = self.build_generator()
        self.generator.summary()

        # High res. and low res. images分别读入高分辨率和低分辨率图像
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # 用低分辨率图像生成假的高分辨率图像
        fake_hr = self.generator(img_lr)

        # 从新生成的HR图像中提取图像特征
        fake_features = self.vgg(fake_hr)

        # 把判别器固定住，只训练生成器
        self.discriminator.trainable = False

        # 判别器对于新生成HR图片的判别结果
        validity = self.discriminator(fake_hr)
        # 输入是高低分辨率图像，输出是判别结果和所提取假图片特征，model包含该输入和输出的全部网络层
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        # self.combined.summary()
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-5, 1],
                              optimizer=optimizer)

    def build_vgg(self):
        """
        采用VGG19的预训练模型来分别从高分辨率图像和生成的fake高分辨率图像中中提取图像特征，
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        print('start loading trianed weights of vgg...')
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        print('loading completes')
        # 只采用VGG的第九层特征
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)

        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """ 残差网络 """
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])  # 把残差边接过来，直接把该残差块的输入和输出短路
            return d

        def grouped_convolution_block(init, grouped_channels, cardinality, strides):
            # grouped_channels 每组的通道数
            # cardinality 多少组
            channel_axis = -1
            group_list = []
            for c in range(cardinality):
                x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(init)
                x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                           kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)  # L2正则化
                group_list.append(x)
            group_merge = concatenate(group_list, axis=channel_axis)
            x = BatchNormalization()(group_merge)
            x = Activation('relu')(x)
            return x

        def block_module(x, filters, cardinality, strides):
            # residual connection
            init = x
            # grouped_channels = int(filters / cardinality)
            grouped_channels = 4
            cardinality = 4
            # 如果没有down sampling就不需要这种操作
            # if init._keras_shape[-1] != 2 * filters:
            #     init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
            #                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            #     init = BatchNormalization()(init)
            # conv1
            x = Conv2D(16, (1, 1), padding='same', use_bias=False,
                       kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # conv2(group)，选择在 group 的时候 down sampling
            x = grouped_convolution_block(x, grouped_channels, cardinality, strides)
            # conv3
            x = Conv2D(256, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = BatchNormalization()(x)

            x = add([init, x])
            x = Activation('relu')(x)
            return x

        def deconv2d(layer_input):
            """上采样+卷积替换反卷积，可以解决反卷积的棋盘格现象"""
            u = UpSampling2D(size=2)(layer_input)  # 把长和宽进行扩张
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)  # 通道数扩展成256维
            u = Activation('relu')(u)
            return u

        # 输入的低分辨率图像128*128*3
        img_lr = Input(shape=self.lr_shape)

        # 残差网络的预处理
        # 第一部分，低分辨率图像进入后会经过一个卷积+RELU函数,用来确保单元匹配
        c1 = Conv2D(256, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # 第二部分，经过16个残差网络结构，每个残差网络内部包含两个卷积+标准化+RELU，还有一个残差边。
        # r = residual_block(c1, self.gf)
        # for _ in range(self.n_residual_blocks - 1):
        #     r = residual_block(r, self.gf)
        x = block_module(c1, 16, 4, 1)
        for _ in range(15):
            x = block_module(x, 16, 4, 1)
            # block moduel set2, we downsampling in the first block module in sets


        # # 残差之后有个汇总特征的网络
        c2 = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # 第三部分，上采样部分，将长宽进行放大，两次上采样后，变为原来的4倍，实现提高分辨率。
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # 通道数为3的卷积，为了变成一个彩色图片
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """判别器结构"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        #  由一堆的卷积+LeakyReLU+BatchNor构成
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)  # 第一层不进行标准化，只进行卷积和激活
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = Dense(self.df * 16)(d8)  # 全连接层，进行1*1的卷积，进行线性组合。
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)
    def scheduler(self,models,epoch):
        # 学习率下降
        if epoch % 20000 == 0 and epoch != 0:
            for model in models:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))

    def train(self, epochs,init_epoch=0, batch_size=1, sample_interval=50):
        d_loss_epoch = []
        g_loss_epoch = []

        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            self.scheduler([self.combined, self.discriminator], epoch)  # 对学习率进行调整
            # if epoch > 30:
            #     sample_interval = 10
            # if epoch > 100:
            #     sample_interval = 50

            # ----------------------
            # 训练判别器
            # ----------------------

            # 读取图像
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # 用低分辨率生成高分辨率图像
            fake_hr = self.generator.predict(imgs_lr)
            # 设置真假图片的标签
            # 判别模型的输出16*16*1.相当于1024个判别结果
            valid = np.ones((batch_size,) + self.disc_patch)  # 维度上直接相加，patchGAN
            fake = np.zeros((batch_size,) + self.disc_patch)

            # 训练判别器网络
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)  # 传入真实图片和标签1实现训练
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)  # 假的图片和标签训练
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  训练生成器网络
            # ------------------

            # 读取数据
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # 用VGG网络提取原高分辨率图像特征
            image_features = self.vgg.predict(imgs_hr)

            # 训练生成器
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            d_loss_epoch.append(d_loss[0])
            g_loss_epoch.append(g_loss[1])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress   计算过程
            # print("%d time: %s" % (epoch, elapsed_time))
            #
            # # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:  # 采样间隔
            #     self.sample_images(epoch)
            # if epoch % 500 == 0 and epoch > 1:
            #     self.generator.save_weights('./saved_model/' + str(epoch) + '.h5')
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, feature loss: %05f] time: %s " \
                % (epoch, epochs,
                     d_loss[0], 100 * d_loss[1],
                     g_loss[1],
                     g_loss[2],
                     elapsed_time))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                # 保存
                if epoch % 500 == 0 and epoch != init_epoch:
                    os.makedirs('weights/%s' % self.dataset_name, exist_ok=True)
                    self.generator.save_weights("weights/%s/gen_epoch%d.h5" % (self.dataset_name, epoch))
                    self.discriminator.save_weights("weights/%s/dis_epoch%d.h5" % (self.dataset_name, epoch))
        self.plot_loss(d_loss_epoch,g_loss_epoch)

    def plot_loss(self,data1,data2):
        plt.plot(data1, c='red')
        plt.plot(data2, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator','Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()

    # def psnr(self,img1, img2):
    #     mse = np.mean((img1 - img2) ** 2)
    #     if mse < 1.0e-10:
    #         return 100
    #     return 10 * math.log10(255.0 ** 2 / mse)

    def psnr(self, img1, img2):
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


    def ssim(self,imageA,imageB):
        imageA = np.array(imageA, dtype=np.uint8)
        imageB = np.array(imageB, dtype=np.uint8)
        # 通道分离，注意顺序BGR不是RGB
        (B1, G1, R1) = cv2.split(imageA)
        (B2, G2, R2) = cv2.split(imageB)
        # convert the images to grayscale BGR2GRAY
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (score0, diffB) = compare_ssim(B1, B2, full=True)
        (score1, diffG) = compare_ssim(G1, G2, full=True)
        (score2, diffR) = compare_ssim(R1, R2, full=True)
        aveScore = (score0 + score1 + score2) / 3
        print("BGR average SSIM: {}".format(aveScore))
        return  aveScore



    def test_images(self, batch_size=1):
        # self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                               img_res=(self.lr_height, self.lr_width))
        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size, is_pred=True)
        os.makedirs('saved_model/', exist_ok=True)
        self.generator.load_weights('./saved_model/w_resnext/gen_epoch' + str(9500) + '.h5')
        fake_hr = self.generator.predict(imgs_lr)
        # ------------------
        #  加入评估指标PSNR
        # ------------------

        r, c = imgs_hr.shape[0], 2
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        # print("行数为%d" %r)
        # print(fake_hr)
        psnr1 = self.psnr(imgs_hr, fake_hr)
        # psnr1 = tf.image.psnr(imgs_hr, fake_hr, max_val=255)
        # ssim1 = tf.image.ssim(imgs_hr,
        #                       fake_hr,
        #                       max_val=255,
        #                       filter_size=256,
        #                       filter_sigma=1.5,
        #                       k1=0.01,
        #                       k2=0.03)
        # with tf.Session() as sess:
        #     psnr = psnr1.eval()
        #     # ssim = ssim1.eval()
        #     print("psnr=%f"%psnr[0])
        #     # print("ssim=%f"% ssim[0])
        print("_______________________________________")
        # Save generated images and the high resolution originals
        # titles = ['Low resolution input', 'Generated Super resolution']
        # fig, axs = plt.subplots(r, c)
        # for row in range(r):
        #     for col, image in enumerate([imgs_lr, fake_hr]):
        #         axs[row, col].imshow(image[row])
        #         axs[row, col].set_title(titles[col])
        #         axs[row, col].axis('off')
        # fig.savefig("./result.png")
        # plt.close()
        fake_hr = np.squeeze(fake_hr)
        imgs_lr = np.squeeze(imgs_lr)
        imgs_hr = np.squeeze(imgs_hr)
        # print(fake_hr)
        # plt.subplot(1, 3, 1)
        # plt.imshow(imgs_lr)
        # plt.axis('off')
        # plt.subplot(1, 3, 2)
        # plt.imshow(imgs_hr)
        # plt.axis('off')
        # plt.subplot(1, 3, 3)
        # # plt.savefig("./lr.png")
        # plt.imshow(fake_hr)
        # plt.axis('off')
        # plt.savefig("./result0.png")
        # plt.show()

        ssim1 = self.ssim(imgs_hr, fake_hr)
        print("这是新的方式：")
        print("psnr=%f" % psnr1)
        print("ssim=%f" % ssim1)
        plt.imshow(imgs_lr)
        plt.axis('off')
        plt.savefig("./saved_model/w_resnext/lr3.png")
        plt.show()

        plt.imshow(imgs_hr)
        plt.axis('off')
        plt.savefig("./saved_model/w_resnext/hr3.png")
        plt.show()

        plt.imshow(fake_hr)
        plt.axis('off')
        # plt.savefig("./fake2.png")
        plt.savefig("./saved_model/w_resnext/fake3.png")

        plt.show()

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
            plt.close()

#
if __name__ == '__main__':
    gan = SRGAN()
    # gan.train(epochs=30000, batch_size=1, sample_interval=50)
    gan.test_images()
