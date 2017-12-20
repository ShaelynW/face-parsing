import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import math
from keras.models import *
from keras.layers import Input, ZeroPadding2D, Conv2DTranspose, Conv2D, MaxPooling2D, Dropout, Cropping2D, Add
from keras.layers import BatchNormalization, Activation, SeparableConv2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import concatenate

from keras.layers.advanced_activations import PReLU
from keras.layers.core import SpatialDropout2D, Permute
from keras.layers.core import Reshape
from keras.layers.convolutional import UpSampling2D

#from crfrnn_layer import CrfRnnLayer
import time

#from data import *
from create_face_data import *
#from resized_data import*
import util
from accuracy import *
#from crfrnn_layer import CrfRnnLayer
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

class myFCN(object):
    def __init__(self, num_class, img_rows=500, img_cols=500 ):
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.num_class=num_class
        
    def load_train_data(self):
        mydata=dataProcess()
        imgs_train, imgs_mask_train=mydata.load_train_data()
        return imgs_train, imgs_mask_train
    
    def load_labeltest_data(self):
        mydata=dataProcess()
        imgs_test, imgs_mask_test=mydata.load_labeltest_data()
        return imgs_test, imgs_mask_test
    
    def load_test_data(self):
        mydata=dataProcess()
        imgs_test=mydata.load_test_data()
        return imgs_test
    
    def load_val_data(self):
        mydata=dataProcess()
        imgs_val, imgs_mask_val=mydata.load_val_data()
        return imgs_val, imgs_mask_val
    
    def get_multi(self):
        img_input=Input((500, 500, 3))
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv0-1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv0-2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv02 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv1 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv2 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(256, (3, 3), padding='same', name='ae_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv3 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(512, (3, 3), padding='same', name='ae_conv4')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2DTranspose(512, (3, 3), strides=(2, 2))(x)
        x = Conv2D(256, (3, 3), padding='same', name='ae_conv5')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv3])
        
        x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv6')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv2])
        
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv7')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv1])
        
        x = Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv8')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv02])
        
        x = Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv9')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        ae_label = Conv2D(self.num_class, (3, 3), padding='same', name='ae_label')(x)
        ae_label = BatchNormalization()(ae_label)
        ae_label = Activation('softmax')(ae_label)
        
        ae_pairwise = Conv2D(1, (3, 3), padding='same', name='ae_pairwise')(x)
        ae_pairwise = BatchNormalization()(ae_pairwise)
        ae_pairwise = Activation('sigmoid')(ae_pairwise)
        
        out = Add()([ae_label, ae_pairwise])

        model = Model(img_input, out)
        return model
    
    def get_shallow_multi(self):
        img_input=Input((500, 500, 3))
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv0-1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv0-2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv02 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv1 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv2 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(256, (3, 3), padding='same', name='ae_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv6')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv2])
        
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv7')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv1])
        
        x = Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv8')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv02])
        
        x = Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv9')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        ae_label = Conv2D(self.num_class, (3, 3), padding='same', name='ae_label')(x)
        ae_label = BatchNormalization()(ae_label)
        ae_label = Activation('softmax')(ae_label)
        
        ae_pairwise = Conv2D(1, (3, 3), padding='same', name='ae_pairwise')(x)
        ae_pairwise = BatchNormalization()(ae_pairwise)
        ae_pairwise = Activation('sigmoid')(ae_pairwise)
        
        out = Add()([ae_label, ae_pairwise])

        model = Model(img_input, out)
        return model
    
    def get_multi_prelu(self):
        img_input=Input((500, 500, 3))
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv0-1')(img_input)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv0-2')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        ae_conv02 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv1')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        ae_conv1 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv2')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        ae_conv2 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(256, (3, 3), padding='same', name='ae_conv3')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        ae_conv3 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(512, (3, 3), padding='same', name='ae_conv4')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        
        x = Conv2DTranspose(512, (3, 3), strides=(2, 2))(x)
        x = Conv2D(256, (3, 3), padding='same', name='ae_conv5')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Add()([x, ae_conv3])
        
        x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv6')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Add()([x, ae_conv2])
        
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv7')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Add()([x, ae_conv1])
        
        x = Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv8')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Add()([x, ae_conv02])
        
        x = Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv9')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        
        ae_label = Conv2D(self.num_class, (3, 3), padding='same', name='ae_label')(x)
        ae_label = BatchNormalization()(ae_label)
        ae_label = Activation('softmax')(ae_label)
        
        ae_pairwise = Conv2D(1, (3, 3), padding='same', name='ae_pairwise')(x)
        ae_pairwise = BatchNormalization()(ae_pairwise)
        ae_pairwise = Activation('sigmoid')(ae_pairwise)
        
        out = Add()([ae_label, ae_pairwise])

        model = Model(img_input, out)
        return model
    
    def get_multi_crf(self, train):
        img_input=Input((500, 500, 3))
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv0-1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv0-2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv02 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv1 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv2 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(256, (3, 3), padding='same', name='ae_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv3 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(512, (3, 3), padding='same', name='ae_conv4')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2DTranspose(512, (3, 3), strides=(2, 2))(x)
        x = Conv2D(256, (3, 3), padding='same', name='ae_conv5')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv3])
        
        x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv6')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv2])
        
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv7')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv1])
        
        x = Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv8')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv02])
        
        x = Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv9')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(self.num_class, (3, 3), padding='same')(x)
        
        if train=="train":
            out = CrfRnnLayer(image_dims=(self.img_rows, self.img_cols),
                              num_classes=self.num_class,
                              theta_alpha=160,
                              theta_beta=3,
                              theta_gamma=3,
                              num_iterations=5,
                              name='crfrnn')([x, img_input])
        else:
            out = CrfRnnLayer(image_dims=(self.img_rows, self.img_cols),
                              num_classes=self.num_class,
                              theta_alpha=160,
                              theta_beta=3,
                              theta_gamma=3,
                              num_iterations=10,
                              name='crfrnn')([x, img_input])
        model = Model(img_input, out)
        return model
    
    def get_test1_multi(self, train):
        img_input=Input((500, 500, 3))
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv0-1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv0-2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv02 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(64, (3, 3), padding='same', name='ae_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv1 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(128, (3, 3), padding='same', name='ae_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv2 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(256, (3, 3), padding='same', name='ae_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv3 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(512, (3, 3), padding='same', name='ae_conv4')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2DTranspose(512, (3, 3), strides=(2, 2))(x)
        x = Conv2D(self.num_class, (3, 3), padding='same')(x)
        ae_conv3 = Conv2D(self.num_class, (3, 3), padding='same')(ae_conv3)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv3])
        
        x = Conv2DTranspose(self.num_class, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(self.num_class, (3, 3), padding='same')(x)
        ae_conv2 = Conv2D(self.num_class, (3, 3), padding='same')(ae_conv2)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv2])
        
        x = Conv2DTranspose(self.num_class, (3, 3), strides=(2, 2))(x)
        x = Conv2D(self.num_class, (3, 3), padding='same')(x)
        ae_conv1 = Conv2D(self.num_class, (3, 3), padding='same')(ae_conv1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv1])
        
        x = Conv2DTranspose(self.num_class, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(self.num_class, (3, 3), padding='same')(x)
        ae_conv02 = Conv2D(self.num_class, (3, 3), padding='same')(ae_conv02)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, ae_conv02])
        
        x = Conv2DTranspose(self.num_class, (1, 1), strides=(2, 2), padding='same')(x)
        x = Conv2D(self.num_class, (3, 3), padding='same', name='ae_label')(x)
        if train=="train":
            out = CrfRnnLayer(image_dims=(self.img_rows, self.img_cols),
                              num_classes=self.num_class,
                              theta_alpha=160,
                              theta_beta=3,
                              theta_gamma=3,
                              num_iterations=5,
                              name='crfrnn')([x, img_input])
        else:
            out = CrfRnnLayer(image_dims=(self.img_rows, self.img_cols),
                              num_classes=self.num_class,
                              theta_alpha=160,
                              theta_beta=3,
                              theta_gamma=3,
                              num_iterations=10,
                              name='crfrnn')([x, img_input])

        model = Model(img_input, out)
        return model
    
    def get_ENet(self):
        def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
            conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
            max_pool = MaxPooling2D()(inp)
            merged = concatenate([conv, max_pool], axis=3)
            return merged
        
        def encoder_bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, 
                       downsample=False, dropout_rate=0.1):
            # main branch
            internal = output // internal_scale        #filers number
            encoder = inp
            # 1x1
            input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
            encoder = Conv2D(internal, (input_stride, input_stride),
                             # padding='same',
                             strides=(input_stride, input_stride), use_bias=False)(encoder)
            # Batch normalization + PReLU
            encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
            encoder = PReLU(shared_axes=[1, 2])(encoder)
            # conv
            if not asymmetric and not dilated:
                encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
            elif asymmetric:
                encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
                encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
            elif dilated:
                encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
            else:
                raise(Exception('You shouldn\'t be here'))
            encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
            encoder = PReLU(shared_axes=[1, 2])(encoder)
            # 1x1
            encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)
            encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
            encoder = SpatialDropout2D(dropout_rate)(encoder)
            other = inp
            # other branch
            if downsample:
                other = MaxPooling2D()(other)
                other = Permute((1, 3, 2))(other)
                pad_feature_maps = output - inp.get_shape().as_list()[3]
                tb_pad = (0, 0)
                lr_pad = (0, pad_feature_maps)
                other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
                other = Permute((1, 3, 2))(other)
            encoder = Add()([encoder, other])
            encoder = PReLU(shared_axes=[1, 2])(encoder)
            return encoder

        def encoder_build(inp, dropout_rate=0.01):
            enet = initial_block(inp)
            enet = BatchNormalization(momentum=0.1)(enet)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
            enet = PReLU(shared_axes=[1, 2])(enet)
            enet = encoder_bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
            for _ in range(4):
                enet = encoder_bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i
            enet = encoder_bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
            # bottleneck 2.x and 3.x
            for _ in range(2):
                enet = encoder_bottleneck(enet, 128)  # bottleneck 2.1
                enet = encoder_bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
                enet = encoder_bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
                enet = encoder_bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
                enet = encoder_bottleneck(enet, 128)  # bottleneck 2.5
                enet = encoder_bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
                enet = encoder_bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
                enet = encoder_bottleneck(enet, 128, dilated=16)  # bottleneck 2.8
            return enet
        
        def decoder_bottleneck(encoder, output, upsample=False, reverse_module=False):
            internal = output // 4
            x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
            x = BatchNormalization(momentum=0.1)(x)
            x = Activation('relu')(x)
            if not upsample:
                x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
            else:
                x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
            x = BatchNormalization(momentum=0.1)(x)
            x = Activation('relu')(x)
            x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)
            other = encoder
            if encoder.get_shape()[-1] != output or upsample:
                other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
                other = BatchNormalization(momentum=0.1)(other)
                if upsample and reverse_module is not False:
                    other = UpSampling2D(size=(2, 2))(other)
            if upsample and reverse_module is False:
                decoder = x
            else:
                x = BatchNormalization(momentum=0.1)(x)
                decoder = Add()([x, other])
                decoder = Activation('relu')(decoder)
            return decoder

        def decoder_build(encoder, nc):
            enet = decoder_bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
            enet = decoder_bottleneck(enet, 64)  # bottleneck 4.1
            enet = decoder_bottleneck(enet, 64)  # bottleneck 4.2
            enet = decoder_bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
            enet = decoder_bottleneck(enet, 16)  # bottleneck 5.1
            enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2),
                                   padding='same')(enet)
            return enet
        
        w = 512
        h = 512
        nc=self.num_class
        data_shape = w * h 
        inp = Input(shape=(500, 500, 3))#1D no
#        inp = Input(shape=(h, w, 3))#1D yes
        inp1 = ZeroPadding2D(padding=(6, 6))(inp)#1D no
        enet = encoder_build(inp1)
        enet = decoder_build(enet, nc)
#        enet = Reshape((data_shape, nc))(enet)  # TODO: need to remove data_shape for multi-scale training
        enet = Reshape((h, w, nc))(enet)#1D no
        enet = Cropping2D((6, 6))(enet)#1D no
        label = Activation('softmax')(enet)
        
        pairwise = Conv2D(1, (3, 3), padding='same')(enet)
        pairwise = BatchNormalization()(pairwise)
        pairwise = Activation('sigmoid')(enet)
        
        out = Add()([label, pairwise])
        model = Model(inp, out)
#        model.compile(optimizer='adam', loss='categorical_crossentropy',
#                      metrics=['accuracy', 'mean_squared_error'])
        return model
    
    def get_MobileNet(self):
        def mobile_block(x, filter_1, filter_2):
            x = SeparableConv2D(filter_1, (3, 3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_1, (1, 1), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = SeparableConv2D(filter_2, (3, 3), strides=(2,2), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_2*2, (1, 1), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        
        def separable_filters(x):
            for i in range(5):
                x = SeparableConv2D(512, kernel_size=(3, 3), strides=(1,1),padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(512, (1,1), strides=(1,1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            return x
        
        def final_conv_block(x):
            x = SeparableConv2D(512, (3, 3), strides=(2,2),padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(1024, (3, 3), strides=(1,1),padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        
        img_input=Input((self.img_rows, self.img_cols, 3))
        x = Conv2D(32, (3, 3), strides=(2,2), padding='same')(img_input)
        pool1 = x
        x = mobile_block(x, 32, 64)          
        pool2 = x
        x = mobile_block(x, 128, 128)
        pool3 = x
        x = mobile_block(x, 256, 256)
        pool4 = x
        x = separable_filters(x)
        x = final_conv_block(x)
        
        up4 = Conv2DTranspose(self.num_class, (2, 2), strides=2, use_bias=False)(x)
        pool4 = Conv2D(self.num_class, (1, 1))(pool4)
        x = Add()([up4, pool4])
        up3 = Conv2DTranspose(self.num_class, (1, 1), strides=2, use_bias=False)(x)
        up3 = Cropping2D(((0, 1), (0, 1)))(up3)
        pool3 = Conv2D(self.num_class, (1, 1))(pool3)
        x = Add()([up3, pool3])
        up2 = Conv2DTranspose(self.num_class, (1, 1), strides=2, use_bias=False)(x)
        up2 = Cropping2D(((0, 1), (0, 1)))(up2)
        pool2 = Conv2D(self.num_class, (1, 1))(pool2)
        x = Add()([up2, pool2])
        up1 = Conv2DTranspose(self.num_class, (2, 2), strides=2, use_bias=False)(x)
        pool1 = Conv2D(self.num_class, (1, 1))(pool1)
        x = Add()([up1, pool1])
        x = Conv2DTranspose(self.num_class, (2, 2), strides=2, use_bias=False)(x)
        
        ae_label = Conv2D(self.num_class, (3, 3), padding='same', name='ae_label')(x)
        ae_label = BatchNormalization()(ae_label)
        ae_label = Activation('softmax')(ae_label)
        
        ae_pairwise = Conv2D(1, (3, 3), padding='same', name='ae_pairwise')(x)
        ae_pairwise = BatchNormalization()(ae_pairwise)
        ae_pairwise = Activation('sigmoid')(ae_pairwise)
        
        out = Add()([ae_label, ae_pairwise])
        model=Model(img_input, out)
        return model
    
    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'
        channel_axis = 3
        
        x = Conv2D(squeeze, (1, 1),  activation='relu', name=s_id + sq1x1)(x)
        
        left = Conv2D(expand, (1, 1), activation='relu', name=s_id + exp1x1)(x)
        
        right = Conv2D(expand, (3, 3), activation='relu', padding='same', name=s_id + exp3x3)(x)
        
        x = Concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
        return x
        
    # Original SqueezeNet from paper.
    def get_SqueezeNet(self):
        def fire_module(x, fire_id, squeeze=16, expand=64): 
            s_id = 'fire' + str(fire_id) + '/'
            channel_axis = 3
            
            x = Conv2D(squeeze, (1, 1),  activation='relu', name=s_id + sq1x1)(x)
            
            left = Conv2D(expand, (1, 1), activation='relu', name=s_id + exp1x1)(x)
            
            right = Conv2D(expand, (3, 3), activation='relu', padding='same', name=s_id + exp3x3)(x)
            
            x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
            
            return x
    
        img_input = Input((self.img_rows, self.img_cols, 3))
        
        x = Conv2D(96, (7, 7), strides=(2, 2), activation='relu', name='conv1')(img_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
        pool1=x
        
        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)        
        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
        pool2=x
        
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)        
        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
        
        
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)    
        x = Dropout(0.5, name='drop9')(x)
        
#        x = Conv2D(1024, (7, 7), activation='relu', padding='valid', name='fc6')(x)#4096
#        x = Dropout(0.5)(x)
        x = Conv2D(1024, (1, 1), activation='relu', padding='valid', name='fc7')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(self.num_class, (1, 1), padding='valid', name='score-fr')(x)
        
        score2 = Conv2DTranspose(self.num_class, (3, 3), strides=2, name='score2')(x)
        score_pool4 = Conv2D(self.num_class, (1, 1), name='score-pool4')(pool2)
#        score_pool4c = Cropping2D((5, 5))(score_pool4)
        score_fused = Add()([score2, score_pool4])
        score4 = Conv2DTranspose(self.num_class, (3, 3), strides=2, name='score4', use_bias=False)(score_fused)
        
        # Skip connections from pool3
        score_pool3 = Conv2D(self.num_class, (1, 1), name='score-pool3')(pool1)
#        score_pool3c = Cropping2D((9, 9))(score_pool3)
        
        # Fuse things together
        score_final = Add()([score4, score_pool3])
        # Final up-sampling and cropping
        upsample = Conv2DTranspose(self.num_class, (3, 3), strides=2, name='upsample', use_bias=False)(score_final)
#        upscore = Cropping2D(((31, 37), (31, 37)))(upsample)

        
        out = Conv2DTranspose(self.num_class, (8, 8), strides=2, name='oute', use_bias=False)(upsample)
       
        x = Activation('softmax', name='loss')(out)
        
        model = Model(img_input, x, name='squeezenet')
        return model
    
    def get_bypass_SqueezeNet(self):
        def fire_module(x, fire_id, squeeze=16, expand=64): 
            s_id = 'fire' + str(fire_id) + '/'
            channel_axis = 3
            
            x = Conv2D(squeeze, (1, 1),  activation='relu', name=s_id + sq1x1)(x)
            
            left = Conv2D(expand, (1, 1), activation='relu', name=s_id + exp1x1)(x)
            
            right = Conv2D(expand, (3, 3), activation='relu', padding='same', name=s_id + exp3x3)(x)
            
            x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
            
            return x
    
        img_input = Input((self.img_rows, self.img_cols, 3))
        
        x = Conv2D(96, (7, 7), strides=(2, 2), activation='relu', name='conv1')(img_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
        pool1=x
        
        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        res1=x
        x = fire_module(x, fire_id=3, squeeze=16, expand=64) 
        x = Add()([res1, x])
        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
        pool2=x
        
        x = fire_module(x, fire_id=5, squeeze=32, expand=128) 
        x = Add()([pool2, x])
        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        res3=x
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = Add()([res3, x])
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
        
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)    
        x = Dropout(0.5, name='drop9')(x)
        
        x = Conv2DTranspose(512, (3, 3), strides=2)(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, pool2])
        
        x = Conv2DTranspose(256, (3, 3), strides=2)(x)
        x = Conv2D(96, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, pool1])
        
        x = Conv2DTranspose(96, (12, 12), strides=4)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
       
        ae_label = Conv2D(self.num_class, (3, 3), padding='same', name='ae_label')(x)
        ae_label = BatchNormalization()(ae_label)
        ae_label = Activation('softmax')(ae_label)
        
        ae_pairwise = Conv2D(1, (3, 3), padding='same', name='ae_pairwise')(x)
        ae_pairwise = BatchNormalization()(ae_pairwise)
        ae_pairwise = Activation('sigmoid')(ae_pairwise)
        
        out = Add()([ae_label, ae_pairwise])
        
        model = Model(img_input, out, name='squeezenet')
        return model
    
    def get_Xception(self):
        img_input=Input((self.img_rows, self.img_cols,3))
        x = ZeroPadding2D(padding=(50, 50))(img_input)
        x=Conv2D(32, (3, 3), strides=(1, 1), use_bias=False, name='block1_conv1')(x)
        x=BatchNormalization(name='block1_conv1_bn')(x)
        x=Activation('relu', name='block1_conv1_act')(x)
        x=Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x=BatchNormalization(name='block1_conv2_bn')(x)
        x=Activation('relu', name='block1_conv2_act')(x)
        
        residual=Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual=BatchNormalization()(residual)
        
        x=SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x=BatchNormalization(name='block2_sepconv1_bn')(x)#para=512
        x=Activation('relu', name='block2_sepconv1_act')(x)
        x=SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x=BatchNormalization(name='block2_sepconv2_bn')(x)#para=512
        block2=x
        x=MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        x=Add()([x,residual])
        
        residual=Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual=BatchNormalization()(residual)
        
        x=Activation('relu',name='block3_sepconv1_act')(x)
        x=SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x=BatchNormalization(name='block3_sepconv1_bn')(x)
        x=Activation('relu', name='block3_sepconv2_act')(x)
        x=SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x=BatchNormalization(name='block3_sepconv2_bn')(x)
        block3=x
        x=MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
        x=Add()([x,residual])
        
        residual=Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual=BatchNormalization()(residual)
        
        x=Activation('relu', name='block4_sepconv1_act')(x)
        x=SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x=BatchNormalization(name='block4_sepconv1_bn')(x)
        x=Activation('relu', name='block4_sepconv2_act')(x)
        x=SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x=BatchNormalization(name='block4_sepconv2_bn')(x)
        block4=x
        x=MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
        x=Add()([x,residual])
        
        for i in range(8):
            residual=x
            prefix='block'+str(i+5)
            
            x=Activation('relu', name=prefix+'_sepconv1_act')(x)
            x=SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix+'_sepconv1')(x)
            x=BatchNormalization(name=prefix+'_sepconv1_bn')(x)
            
            x=Activation('relu', name=prefix+'_sepconv2_act')(x)
            x=SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix+'_sepconv2')(x)
            x=BatchNormalization(name=prefix+'_sepconv2_bn')(x)
            
            x=Activation('relu',name=prefix+'_sepconv3_act')(x)
            x=SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix+'_sepconv3')(x)
            x=BatchNormalization(name=prefix+'_sepconv3_bn')(x)
            
            x=Add()([x,residual])
            
        residual=Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual=BatchNormalization()(residual)
        
        x=Activation('relu', name='block13_sepconv1_act')(x)
        x=SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
        x=BatchNormalization(name='block13_sepconv1_bn')(x)
        
        x=Activation('relu', name='block13_sepconv2_act')(x)
        x=SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
        x=BatchNormalization(name='block13_sepconv2_bn')(x)
        block13=x
        
        x=MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
        x=Add()([x, residual])
        
        x=SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
        x=BatchNormalization(name='block14_sepconv1_bn')(x)
        x=Activation('relu', name='block14_sepconv1_act')(x)
        
        x=SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
        x=BatchNormalization(name='block14_sepconv2_bn')(x)
        x=Activation('relu', name='block14_sepconv2_act')(x)
        
        x = Conv2D(2048, (1, 1), activation='relu', padding='valid', name='fc7')(x)
        x = Dropout(0.5)(x)
        x=Conv2D(self.num_class, (1, 1), padding='valid', name='up0')(x)
        
        up1=Conv2DTranspose(self.num_class, (2, 2), strides=4, name='up1')(x)
        up1=Cropping2D(((1,1),(1,1)))(up1)
        up_block4=Conv2D(self.num_class,(1, 1),name='up_block4')(block4)
        up_block13=Conv2DTranspose(self.num_class, (2, 2), strides=2, name='up_block13')(block13)
        fuse1=Add()([up1, up_block4, up_block13])
        
        up2=Conv2DTranspose(self.num_class, (2, 2), strides=2, name='up2')(fuse1)
        up_block3=Conv2D(self.num_class,(1, 1),name='up_block3')(block3)
        fuse2=Add()([up2, up_block3])
        
        up3=Conv2DTranspose(self.num_class, (2, 2), strides=2)(fuse2)
        up_block2=Conv2D(self.num_class,(1, 1),name='up_block2')(block2)
        fuse3=Add()([up3, up_block2])
        
        out=Cropping2D(((48,48),(48,48)))(fuse3)
        model=Model(img_input,out)
        return model
    

    def get_FCN_crf(self, train):
        img_input=Input((self.img_rows, self.img_cols, 3))
        # Add plenty of zero padding
        x = ZeroPadding2D(padding=(100, 100))(img_input)
        
        # VGG-16 convolution block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # VGG-16 convolution block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding='same')(x)
        
        # VGG-16 convolution block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(x)
        pool3 = x
        
        # VGG-16 convolution block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same')(x)
        pool4 = x
        
        # VGG-16 convolution block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', padding='same')(x)
        
        # Fully-connected layers converted to convolution layers
        x = Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6')(x)#4096
        x = Dropout(0.5)(x)
        x = Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(self.num_class, (1, 1), padding='valid', name='score-fr')(x)
        
        # Deconvolution
        score2 = Conv2DTranspose(self.num_class, (4, 4), strides=2, name='score2')(x)
        
        # Skip connections from pool4
        score_pool4 = Conv2D(self.num_class, (1, 1), name='score-pool4')(pool4)
        score_pool4c = Cropping2D((5, 5))(score_pool4)
        score_fused = Add()([score2, score_pool4c])
        score4 = Conv2DTranspose(self.num_class, (4, 4), strides=2, name='score4', use_bias=False)(score_fused)
        
        # Skip connections from pool3
        score_pool3 = Conv2D(self.num_class, (1, 1), name='score-pool3')(pool3)
        score_pool3c = Cropping2D((9, 9))(score_pool3)
        
        # Fuse things together
        score_final = Add()([score4, score_pool3c])
        # Final up-sampling and cropping
        upsample = Conv2DTranspose(self.num_class, (16, 16), strides=8, name='upsample', use_bias=False)(score_final)
        x = Cropping2D(((31, 37), (31, 37)))(upsample)
        
        if train=="train":
            out = CrfRnnLayer(image_dims=(self.img_rows, self.img_cols),
                              num_classes=self.num_class,
                              theta_alpha=160,
                              theta_beta=3,
                              theta_gamma=3,
                              num_iterations=5,
                              name='crfrnn')([x, img_input])
        else:
            out = CrfRnnLayer(image_dims=(self.img_rows, self.img_cols),
                              num_classes=self.num_class,
                              theta_alpha=160,
                              theta_beta=3,
                              theta_gamma=3,
                              num_iterations=10,
                              name='crfrnn')([x, img_input])
        '''
        output = CrfRnnLayer(image_dims=(self.img_rows, self.img_cols),
                         num_classes=self.num_class,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=5,
                         name='crfrnn')([upscore, img_input])
        
        model = Model(img_input, output, name='crfrnn_net')
        '''
        model=Model(img_input, out)
#        for layer in model.layers[:34]:
#            layer.trainable=False
#       
        return model
    
    def get_FCN(self):
        img_input=Input((self.img_rows, self.img_cols, 3))
        # Add plenty of zero padding
        x = ZeroPadding2D(padding=(100, 100))(img_input)
        
        # VGG-16 convolution block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # VGG-16 convolution block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding='same')(x)
        
        # VGG-16 convolution block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(x)
        pool3 = x
        
        # VGG-16 convolution block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same')(x)
        pool4 = x
        
        # VGG-16 convolution block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', padding='same')(x)
        
        # Fully-connected layers converted to convolution layers
        x = Conv2D(2048, (7, 7), activation='relu', padding='valid', name='fc6')(x)#4096
        x = Dropout(0.5)(x)
        x = Conv2D(2048, (1, 1), activation='relu', padding='valid', name='fc7')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(self.num_class, (1, 1), padding='valid', name='score-fr')(x)
        
        # Deconvolution
        score2 = Conv2DTranspose(self.num_class, (4, 4), strides=2, name='score2')(x)
        
        # Skip connections from pool4
        score_pool4 = Conv2D(self.num_class, (1, 1), name='score-pool4')(pool4)
        score_pool4c = Cropping2D((5, 5))(score_pool4)
        score_fused = Add()([score2, score_pool4c])
        score4 = Conv2DTranspose(self.num_class, (4, 4), strides=2, name='score4', use_bias=False)(score_fused)
        
        # Skip connections from pool3
        score_pool3 = Conv2D(self.num_class, (1, 1), name='score-pool3')(pool3)
        score_pool3c = Cropping2D((9, 9))(score_pool3)
        
        # Fuse things together
        score_final = Add()([score4, score_pool3c])
        # Final up-sampling and cropping
        upsample = Conv2DTranspose(self.num_class, (16, 16), strides=8, name='upsample', use_bias=False)(score_final)
        upscore = Cropping2D(((31, 37), (31, 37)))(upsample)
        
#        ae_label = Conv2D(self.num_class, (3, 3), padding='same', name='ae_label')(upscore)
#        ae_label = BatchNormalization()(ae_label)
#        ae_label = Activation('softmax')(ae_label)
#        
#        ae_pairwise = Conv2D(1, (3, 3), padding='same', name='ae_pairwise')(upscore)
#        ae_pairwise = BatchNormalization()(ae_pairwise)
#        ae_pairwise = Activation('sigmoid')(ae_pairwise)
#        
#        out = Add()([ae_label, ae_pairwise])
        
        '''
        output = CrfRnnLayer(image_dims=(self.img_rows, self.img_cols),
                         num_classes=self.num_class,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=5,
                         name='crfrnn')([upscore, img_input])
        
        model = Model(img_input, output, name='crfrnn_net')
        '''
#        model=Model(img_input, [ae_label, ae_pairwise, out])
        model=Model(img_input, upscore)
#       
        return model
    
    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train=self.load_train_data()
        imgs_val, imgs_mask_val=self.load_val_data()
        print("loading data done")
        """
        print('generate data')
        datagen=ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                  rotation_range=20, width_shift_range=0.2,
                                  height_shift_range=0.2, horizontal_flip=True)
        datagen.fit(imgs_train)
        print('finish generation') 
        """
        model = self.get_multi()#nice
#        model=self.get_bypass_SqueezeNet()#nice
#        model = self.get_simplified_multi()#no good, cells
#        model=self.get_test_multi()#no good, without sigmoid
#        model=self.get_test1_multi("train")#very bad
#        model=self.get_multi_crf("train")#very bad
#        model=self.get_FCN()   #error

#        model=self.get_FCN_crf("train")  #load vgg16 model, layers before trainable=False,all black,maybe load no model for train
#        model=self.get_FCN_crf("train")  #load my model, layers before trainable=True 
#        model=self.get_shallow_multi() #bad
#        model=self.get_multi_prelu()


#        model=self.get_ENet()
#        model = self.get_MobileNet()

#        model=self.get_softmax_FCN()
#        model=self.get_SqueezeNET()
#        model=self.get_BatchNormalized_FCN()
#        model=self.get_atrous_FCN()
#        model=self.get_Xception()
        print("got model")
        
        def dice_coef(y_true,y_pred):
            y_true_f=K.flatten(y_true)
            y_pred_f=K.flatten(y_pred)
            intersection=K.sum(y_true_f*y_pred_f)
            dice=(2.*intersection+1)/(K.sum(y_true_f)+K.sum(y_pred_f)+1)
            return dice
        
        def dice_coef_loss(y_true,y_pred):
            return -dice_coef(y_true,y_pred)
        
        def binary_crossentropy_with_logits(y_true,y_pred):#good
            return K.mean(K.binary_crossentropy(y_true,y_pred, from_logits=True), axis=-1)
        
        def mse_flatten(y_true,y_pred):
            y_true_f=K.flatten(y_true)
            y_pred_f=K.flatten(y_pred)
            return K.mean(K.square(y_pred_f-y_true_f), axis=-1)
        
#        model.compile(optimizer=Adam(lr=1e-4),loss='mean_absolute_error',metrics=['binary_accuracy'])#unsuccessful
#        model.compile(optimizer=Adam(lr=1e-4),loss='mean_absolute_percentage_error',metrics=['binary_accuracy'])#unsuccessful
#        model.compile(optimizer=Adam(lr=1e-4),loss='mean_squared_logarithmic_error',metrics=['binary_accuracy'])#similar to mse12
#        model.compile(optimizer=Adam(lr=1e-4),loss='squared_hinge',metrics=['binary_accuracy'])#unsuccessful
#        model.compile(optimizer=Adam(lr=1e-4),loss='hinge',metrics=['binary_accuracy'])
#        model.compile(optimizer=Adam(lr=1e-4),loss='categorical_hinge',metrics=['binary_accuracy'])
#        model.compile(optimizer=Adam(lr=1e-4),loss='logcosh',metrics=['binary_accuracy'])#similar to mse12
#        model.compile(optimizer=Adam(lr=1e-4),loss='cosine_proximity',metrics=['binary_accuracy'])#good
#        model.compile(optimizer=Adam(lr=1e-4),loss='kullback_leibler_divergence',metrics=['binary_accuracy'])#unsuccessful
#        model.compile(optimizer=Adam(lr=1e-4),loss='poisson',metrics=['binary_accuracy'])#unsuccessful
#        model.compile(optimizer=Adam(lr=1e-4),loss='mean_squared_error',metrics=['binary_accuracy'])#good
#        model.compile(optimizer=Adam(lr=1e-4),loss=binary_crossentropy_with_logits, metrics=['binary_accuracy'])#good,but coarse no better than mse
#
        model.compile(optimizer=Adam(lr=1e-4),loss=mse_flatten, metrics=['binary_accuracy'])#better than mse
#        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
#        model.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

#        model.compile(optimizer=Adam(lr=1e-4),loss=dice_coef_loss, metrics=['binary_accuracy'])#epoch=7, no good
#        model.compile(optimizer=SGD(lr=0.04, momentum=0.9,nesterov=True),loss=mse_flatten, metrics=['binary_accuracy'])#for SqueezeNet, unsuccessful
        
                      
#        model.compile(optimizer=SGD(lr=1e-4,momentum=0.9, decay=5**-4, nesterov=True), loss='mean_squared_error',metrics=['binary_accuracy'])
#        model.compile(optimizer=SGD(lr=1e-13,momentum=0.99,nesterov=True),loss='binary_crossentropy',metrics=['accuracy'])#interrupt unsuccessful

        
#        model.load_weights("/home/boriska/Downloads/myFCN/my_FCN/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
        model_checkpoint=ModelCheckpoint('myFCN.h5',monitor='val_loss',verbose=1,save_best_only=True)
        lrate=ReduceLROnPlateau(factor=0.1,verbose=1, patience=5, min_lr=1e-13)#default monitor=val_loss,patience=10(epochs)
        batch_size=2
        tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
#        print("fitting model...")
#        history=model.fit(imgs_train, imgs_mask_train, batch_size=2,epochs=10,verbose=1,
#                  validation_split=0.2, shuffle=True, callbacks=[model_checkpoint,lrate])

#        model.load_weights('myFCN.h5')
        history=model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=100, verbose=1,
                  validation_data=(imgs_val,imgs_mask_val), shuffle=True, callbacks=[model_checkpoint,lrate,tensor_board],
                initial_epoch=0)#initial_epoch=10, start from 11
        
        
        
#        history=model.fit_generator(datagen.flow(imgs_train, imgs_mask_train, batch_size=2,
#                                                 save_to_dir="gen", save_prefix='', save_format='png'),
#                            steps_per_epoch=len(imgs_train), epochs=10, verbose=1, 
#                           validation_data=(imgs_val,imgs_mask_val), callbacks=[model_checkpoint,lrate])
        
        ##################################MSRC##################################
#        train_file_path='/home/boriska/Downloads/MSRC_image/Train.txt'
#        data_dir='/home/boriska/Downloads/MSRC_image/Images'
#        label_dir='/home/boriska/Downloads/MSRC_image/GroundTruth'
#        target_size=(500, 500)
#        classes=21
#        train_datagen=util.SegDataGenerator(zoom_range=[0.5,2.0], zoom_maintain_shape=True,
#                                       crop_mode='random', crop_size=target_size,
#                                       rotation_range=0., shear_range=0, horizontal_flip=True,
#                                       channel_shift_range=20., fill_mode='constant',
#                                       label_cval=0)
#        history=model.fit_generator(generator=train_datagen.flow_from_directory(
#                file_path=train_file_path, data_dir=data_dir, data_suffix='bmp',
#                label_dir=label_dir, label_suffix='bmp', classes=classes,
#                target_size=target_size, color_mode='rgb', batch_size=2,
#                shuffle=True, save_to_dir="gen", save_prefix='', save_format='bmp',
#                loss_shape=(target_size[0]*target_size[1]*classes,),
#                ignore_label=255),
#                steps_per_epoch=len(imgs_train), epochs=10, verbose=1, 
#                validation_data=(imgs_val,imgs_mask_val), callbacks=[model_checkpoint,lrate])
       ########################################################################## 
        
        
        """
        import matplotlib.pyplot as plt
        print(history.history.keys())#list all data in history
        #summarize history for accuracy
        plt.plot(history.history['binary_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'],loc='upper left')
        plt.show()
        #summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','val'],loc='upper right')
        plt.show()
        """
        
    def predict(self):
        print("predict test data")
        imgs_test=self.load_test_data()
        
#        model=self.get_softmax_FCN()
        model = self.get_multi()
#        model = self.get_MobileNet()
#        model=self.get_bypass_SqueezeNet()
#        model=self.get_test_multi()
#        model = self.get_simplified_multi()
#        model=self.get_test1_multi("train")
#        model=self.get_FCN()
#        model=self.get_shallow_multi()
#        model=self.get_multi_prelu()
#        model=self.get_FCN_crf("train")
#        model=self.get_multi_crf("test")
#        model=self.get_ENet()

#        model=self.get_SqueezeNET()
#        model=self.get_BatchNormalized_FCN()
#        model=self.get_atrous_FCN()
#        model=self.get_Xception()

        model.load_weights('myFCN.h5')
        start = time.clock()
        imgs_mask_test, _, _=model.predict(imgs_test, batch_size=1, verbose=1)
        end = time.clock()
        save_dir=os.path.join(os.getcwd(),"img_result")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.getcwd() + "/test_image_name.txt") as f:
            txt = f.readlines()
            txt=[line.split(' ') for line in txt]
        for i in range(len(txt)):
            segmentation = util.get_label_image(imgs_mask_test[i,:,:,:], int(txt[i][1]),int(txt[i][2]))
#            segmentation = util.get_eNet_label_image(imgs_mask_test[i,:,:], int(txt[i][1]),int(txt[i][2]))
            segmentation.save(os.path.join(save_dir,txt[i][0][:-4]+".png"))  
            print('.',end='')
        print("Test end")
        print('predict time: {}s'.format(end-start))
    
    def evaluate(self):
        print("evaluate test data")
        imgs_test, imgs_mask=self.load_labeltest_data()
        print("loading data done")
        
#        model=self.get_softmax_FCN()
        model = self.get_multi()
#        model = self.get_MobileNet()
#        model=self.get_bypass_SqueezeNet()
#        model=self.get_test_multi()
#        model = self.get_simplified_multi()
#        model=self.get_test1_multi("train")
#        model=self.get_FCN()
#        model=self.get_shallow_multi()
#        model=self.get_multi_prelu()
#        model=self.get_FCN_crf("train")
#        model=self.get_multi_crf("test")
#        model=self.get_ENet()

#        model=self.get_SqueezeNET()
#        model=self.get_BatchNormalized_FCN()
#        model=self.get_atrous_FCN()
#        model=self.get_Xception()

        model.load_weights('myFCN.h5')
        start = time.clock()
        imgs_mask_test=model.predict(imgs_test, batch_size=1, verbose=1)
        end = time.clock()
        print('predict time: {}s'.format(end-start))
        save_dir=os.path.join(os.getcwd(),"img_result")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.getcwd() + "/test_image_name.txt") as f:
            txt = f.readlines()
            txt=[line.split(' ') for line in txt]
        pa_list = []
        ma_list = []
        m_IU_list = []
        fw_IU_list = []
        for i in range(len(txt)):
            segmentation = util.get_label_image(imgs_mask_test[i,:,:,:], int(txt[i][1]),int(txt[i][2]))
#            segmentation = util.get_sig_image(imgs_mask_test[i,:,:], int(txt[i][1]),int(txt[i][2]))
            segmentation.save(os.path.join(save_dir,txt[i][0][:-4]+".png"))  
#            cv2.imwrite(os.path.join(save_dir,txt[i][0][:-4]+".png"),segmentation)
            print('.',end='')
            
            seg = imgs_mask_test[i,:,:,:].argmax(axis=2).astype("uint8")[:int(txt[i][1]), :int(txt[i][2])]
            mask = imgs_mask[i,:,:][:int(txt[i][1]), :int(txt[i][2])]
            pa = pixel_accuracy(seg, mask)
            ma = mean_accuracy(seg, mask)
            m_IU = mean_IU(seg, mask)
            fw_IU = frequency_weighted_IU(seg, mask)
            pa_list.append(pa)
            ma_list.append(ma)
            m_IU_list.append(m_IU)
            fw_IU_list.append(fw_IU)
        print("Test evaultate end")
        print("pixel_accuracy: "+str(np.mean(pa_list)))
        print("mean_accuracy: "+str(np.mean(ma_list)))
        print("mean_IU: "+str(np.mean(m_IU_list)))
        print("frequency_weighted: "+str(np.mean(fw_IU_list)))
        
    def getModel(self):
#        model=self.get_SqueezeNet()
#        model=self.get_bypass_SqueezeNet()
#        model=self.get_multi_crf("train")
        model=self.get_FCN()
#        model=self.get_MobileNet()
        model.summary()
        
if __name__=='__main__':
    myFCN=myFCN(num_class=11)
#    myFCN.getModel()
    myFCN.train()
#    myFCN.predict()
    myFCN.evaluate()