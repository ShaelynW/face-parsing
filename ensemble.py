import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import math
from keras.models import *
from keras.layers import Input, ZeroPadding2D, Conv2DTranspose, Conv2D, MaxPooling2D, Dropout, Cropping2D, Add, Average
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
    
    def get_multi(self, img_input):
#        img_input=Input((500, 500, 3))
        x = Conv2D(16, (3, 3), padding='same', name='ae_conv0-1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(32, (3, 3), padding='same', name='ae_conv0-2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        ae_conv02 = x
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(64, (5, 5), padding='same', name='ae_conv1')(x)
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
    
    def get_bypass_SqueezeNet(self, img_input):
        def fire_module(x, fire_id, squeeze=16, expand=64): 
            s_id = 'fire' + str(fire_id) + '/'
            channel_axis = 3
            
            x = Conv2D(squeeze, (1, 1),  activation='relu', name=s_id + sq1x1)(x)
            
            left = Conv2D(expand, (1, 1), activation='relu', name=s_id + exp1x1)(x)
            
            right = Conv2D(expand, (3, 3), activation='relu', padding='same', name=s_id + exp3x3)(x)
            
            x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
            
            return x
    
#        img_input = Input((self.img_rows, self.img_cols, 3))
        
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
       
        ae_label = Conv2D(self.num_class, (3, 3), padding='same')(x)
        ae_label = BatchNormalization()(ae_label)
        ae_label = Activation('softmax')(ae_label)
        
        ae_pairwise = Conv2D(1, (3, 3), padding='same')(x)
        ae_pairwise = BatchNormalization()(ae_pairwise)
        ae_pairwise = Activation('sigmoid')(ae_pairwise)
        
        out = Add()([ae_label, ae_pairwise])
        
        model = Model(img_input, out, name='squeezenet')
        return model
    
    def get_FCN(self, img_input):
#        img_input=Input((self.img_rows, self.img_cols, 3))
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
        
        model=Model(img_input, upscore)
#       
        return model
        
    def evaluate(self):
        def ensemble(models, model_input):
            outputs = [model.outputs[0] for model in models]
            y = Average()(outputs)        
            model = Model(model_input, y)
            return model
        
        print("evaluate test data")
        imgs_test, imgs_mask=self.load_labeltest_data()
        print("loading data done")
        
        img_input=Input((self.img_rows, self.img_cols, 3))
        multi_model = self.get_multi(img_input)
        FCN_model = self.get_FCN(img_input)
        SqueezeNet_model = self.get_bypass_SqueezeNet(img_input)
        
        multi_model.load_weights('/home/boriska/Downloads/myFCN/my_FCN/face_100_multi/face_msef_100_multi.h5')
        FCN_model.load_weights('/home/boriska/Downloads/myFCN/my_FCN/face_50/face_msef_50.h5')
        SqueezeNet_model.load_weights('/home/boriska/Downloads/myFCN/my_FCN/face_100_bypassSqueeze/face_msef_100_bypassSqueeze.h5')
        
        models = [multi_model, FCN_model, SqueezeNet_model]        
        model = ensemble(models, img_input)
        
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
        
if __name__=='__main__':
    myFCN=myFCN(num_class=11)
    myFCN.evaluate()