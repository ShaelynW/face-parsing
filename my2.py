import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import time
from keras.models import *
from keras.layers import Input, ZeroPadding2D, Conv2DTranspose, Conv2D, MaxPooling2D, Dropout, Cropping2D, Add
from keras.layers import BatchNormalization, Activation, SeparableConv2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.preprocessing.image import array_to_img
import cv2

#from crfrnn_layer import CrfRnnLayer
from create_face_data import *
from create_edge_data import *
import util
from accuracy import *

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
    
    def load_train_edge(self):
        mydata=edgeProcess()
        imgs=mydata.load_train_data()
        return imgs
    
    def load_test_edge(self):
        mydata=edgeProcess()
        imgs=mydata.load_test_data()
        return imgs
    
    def load_val_edge(self):
        mydata=edgeProcess()
        imgs=mydata.load_val_data()
        return imgs
    
    def get_multi(self):
        img_input=Input((self.img_rows, self.img_cols, 3))
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

        model = Model(img_input, [out, ae_pairwise])
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
    
    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train=self.load_train_data()
        imgs_val, imgs_mask_val=self.load_val_data()
        train_edge=self.load_train_edge()
        val_edge=self.load_val_edge()
        print("loading data done")
       
        model = self.get_multi()#nice
#        model=self.get_multi_crf("train")#very bad
        print("got model")
        
        def mse_flatten(y_true,y_pred):
            y_true_f=K.flatten(y_true)
            y_pred_f=K.flatten(y_pred)
            return K.mean(K.square(y_pred_f-y_true_f), axis=-1)
        
        model.compile(optimizer=Adam(lr=1e-4),loss=mse_flatten, metrics=['binary_accuracy'])#better than mse
#        model.compile(optimizer=SGD(lr=0.04, momentum=0.9,nesterov=True),loss=mse_flatten, metrics=['binary_accuracy'])#for SqueezeNet, unsuccessful
        
        model_checkpoint=ModelCheckpoint('myFCN.h5',monitor='val_loss',verbose=1,save_best_only=True)
        lrate=ReduceLROnPlateau(factor=0.1,verbose=1, patience=5, min_lr=1e-13)#default monitor=val_loss,patience=10(epochs)
        batch_size=2
        tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
        print("fitting model...")
#        history=model.fit(imgs_train, imgs_mask_train, batch_size=2,epochs=10,verbose=1,
#                  validation_split=0.2, shuffle=True, callbacks=[model_checkpoint,lrate])

#        model.load_weights('myFCN.h5')
        history=model.fit(imgs_train, [imgs_mask_train, train_edge], batch_size=batch_size, 
                          epochs=100, verbose=1, validation_data=(imgs_val,[imgs_mask_val, val_edge]),
                          shuffle=True, callbacks=[model_checkpoint,lrate,tensor_board],
                          initial_epoch=0)#initial_epoch=10, start from 11  
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
        
        model = self.get_multi()
#        model=self.get_multi_crf("test")

        model.load_weights('myFCN.h5')
        start = time.clock()
        imgs_mask_test, edge=model.predict(imgs_test, batch_size=1, verbose=1)
        end = time.clock()
        save_dir=os.path.join(os.getcwd(),"img_result")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.getcwd() + "/test_image_name.txt") as f:
            txt = f.readlines()
            txt=[line.split(' ') for line in txt]
        for i in range(len(txt)):
            segmentation = util.get_label_image(imgs_mask_test[i,:,:,:], int(txt[i][1]),int(txt[i][2]))
            segmentation.save(os.path.join(save_dir,txt[i][0][:-4]+".png"))  
            print('.',end='')
        print("Test end")
        print('predict time: {}s'.format(end-start))
    
    def evaluate(self):
        print("evaluate test data")
        imgs_test, imgs_mask=self.load_labeltest_data()
        print("loading data done")
        
        model = self.get_multi()
#        model=self.get_multi_crf("test")

        model.load_weights('myFCN.h5')
        start = time.clock()
        imgs_mask_test, edge=model.predict(imgs_test, batch_size=1, verbose=1)
        end = time.clock()
        print('predict time: {}s'.format(end-start))
        save_dir=os.path.join(os.getcwd(),"img_result")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        edge_save_dir=os.path.join(os.getcwd(),"edge_result")
        if not os.path.isdir(edge_save_dir):
            os.makedirs(edge_save_dir)
        
        with open(os.getcwd() + "/test_image_name.txt") as f:
            txt = f.readlines()
            txt=[line.split(' ') for line in txt]
        pa_list = []
        ma_list = []
        m_IU_list = []
        fw_IU_list = []
        for i in range(len(txt)):
            segmentation = util.get_label_image(imgs_mask_test[i,:,:,:], int(txt[i][1]),int(txt[i][2]))
            segmentation.save(os.path.join(save_dir,txt[i][0][:-4]+".png"))  
#            
#            edge = util.get_sig_image(edge[i,:,:], int(txt[i][1]),int(txt[i][2]))
#            cv2.imwrite(os.path.join(edge_save_dir,txt[i][0][:-4]+".png"), edge)
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
        model=self.get_multi_crf("train")
        model.summary()
        
if __name__=='__main__':
    myFCN=myFCN(num_class=11)
#    myFCN.getModel()
#    myFCN.train()
#    myFCN.predict()
    myFCN.evaluate()