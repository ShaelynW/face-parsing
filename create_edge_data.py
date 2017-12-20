import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img
import util

class edgeProcess(object):
    def __init__(self, train_label_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/train_edge",
                 test_label_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/test_edge",
                 val_label_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/val_edge",
                 npy_path=os.getcwd()):
        self.train_label_path=train_label_path
        self.test_label_path=test_label_path
        self.val_label_path=val_label_path
        self.npy_path=npy_path
        
    def create_train_data(self):
        imglabels=[]
        file_path=self.train_label_path
        imgs=os.listdir(file_path)#2134243149_2.png
        for i in range(len(imgs)):
            label_path=os.path.join(file_path, imgs[i])
            label,y_h,y_w=util.get_edge(label_path)
            imglabels.append(label)
            if i%100 ==0:
                print('Done:{0}/{1} images'.format(i,len(imgs)))
        imglabels=np.array(imglabels,dtype=np.uint8)
        print("loading done")
        np.save(self.npy_path+'/edge_train.npy',imglabels)
        print('Saving to npy files done.')
    
    def create_val_data(self):
        imglabels=[]
        file_path=self.val_label_path
        imgs=os.listdir(file_path)#2134243149_2.png
        for i in range(len(imgs)):
            label_path=os.path.join(file_path, imgs[i])
            label,y_h,y_w=util.get_edge(label_path)
            imglabels.append(label)
            if i%100 ==0:
                print('Done:{0}/{1} images'.format(i,len(imgs)))
        imglabels=np.array(imglabels,dtype=np.uint8)
        print("loading done")
        np.save(self.npy_path+'/edge_val.npy',imglabels)
        print('Saving to npy files done.')
        
    def create_test_data(self):
        imglabels=[]
        file_path=self.test_label_path
        imgs=os.listdir(file_path)#2134243149_2.png
        for i in range(len(imgs)):
            label_path=os.path.join(file_path, imgs[i])
            label,y_h,y_w=util.get_edge(label_path)
            imglabels.append(label)
            if i%100 ==0:
                print('Done:{0}/{1} images'.format(i,len(imgs)))
        imglabels=np.array(imglabels,dtype=np.uint8)
        print("loading done")
        np.save(self.npy_path+'/edge_test.npy',imglabels)
        print('Saving to npy files done.')
        
    def load_train_data(self):
        imgs_train=np.load(self.npy_path+"/edge_train.npy")
        imgs_train=imgs_train.astype('float32')
        imgs_train/=255   #0,255->0,1
        imgs_train=imgs_train.astype('uint8')
        return imgs_train
    
    def load_test_data(self):
        imgs_test=np.load(self.npy_path+"/edge_test.npy") 
        imgs_test=imgs_test.astype('float32')
        imgs_test/=255   #0,255->0,1
        imgs_test=imgs_test.astype('uint8')
        return imgs_test
    
    def load_val_data(self):
        imgs_val=np.load(self.npy_path+"/edge_val.npy")
        imgs_val=imgs_val.astype('float32')
        imgs_val/=255   #0,255->0,1
        imgs_val=imgs_val.astype('uint8')
        return imgs_val
    
if __name__=="__main__":
    mydata=edgeProcess() 
    mydata.create_train_data()
    mydata.create_test_data()
    mydata.create_val_data()