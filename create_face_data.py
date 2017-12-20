import numpy as np
import os
import glob
from keras.preprocessing.image import img_to_array, load_img

import util

class dataProcess(object):
    def __init__(self, data_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/images",
                 label_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/index",
                 test_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/test",
                 test_label_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/testannot",
                 val_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/val",
                 val_label_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/valannot",
                 npy_path=os.getcwd(),
                 img_type=".jpg",
                 annot_img_type=".png"):
        self.data_path=data_path
        self.label_path=label_path
        self.test_path=test_path
        self.test_label_path=test_label_path
        self.val_path=val_path
        self.val_label_path=val_label_path
        self.npy_path=npy_path
        self.img_type=img_type
        self.annot_img_type=annot_img_type
        
    def create_train_data(self):
        i=0
        imgdatas=[]
        imglabels=[]
        imgs=glob.glob(self.data_path+"/*"+self.img_type)#get a list of images' full names(including path)
        for imgname in imgs:
            midname=imgname[imgname.rindex("/")+1:-4]#get image name
#            img=img_to_array(load_img(self.data_path+"/"+midname))
            img,img_h,img_w=util.get_image(self.data_path+"/"+midname+self.img_type) 
#            label=img_to_array(load_img(self.label_path+"/"+midname))
            label,y_h,y_w=util.get_label(self.label_path+"/"+midname+self.annot_img_type)
            imgdatas.append(img)
            imglabels.append(label)
            if i%100 ==0:
                print('Done:{0}/{1} images'.format(i,len(imgs)))
            i+=1
        imgdatas=np.array(imgdatas,dtype=np.uint8)
        imglabels=np.array(imglabels,dtype=np.uint8)
        print("loading done")
        np.save(self.npy_path+'/imgs_train.npy',imgdatas)
        np.save(self.npy_path+'/imgs_mask_train.npy',imglabels)
        print('Saving to npy files done.')
        
    def create_labeltest_data(self):
        i=0
        imgdatas=[]
        imglabels=[]
        imgs=glob.glob(self.test_path+"/*"+self.img_type)#get a list of images' full names(including path)
        for imgname in imgs:
            midname=imgname[imgname.rindex("/")+1:-4]#get image name
#            img=img_to_array(load_img(self.data_path+"/"+midname))
            img,img_h,img_w=util.get_image(self.test_path+"/"+midname+self.img_type) 
#            label=img_to_array(load_img(self.label_path+"/"+midname))
            label,y_h,y_w=util.get_test_label(self.test_label_path+"/"+midname+self.annot_img_type)
            imgdatas.append(img)
            imglabels.append(label)
            if i%100 ==0:
                print('Done:{0}/{1} images'.format(i,len(imgs)))
            i+=1
        imgdatas=np.array(imgdatas,dtype=np.uint8)
        imglabels=np.array(imglabels,dtype=np.uint8)
        print("loading done")
        np.save(self.npy_path+'/imgs_labeltest.npy',imgdatas)
        np.save(self.npy_path+'/imgs_mask_test.npy',imglabels)
        print('Saving to npy files done.')
        
    def create_test_data(self):
        i=0
        f=open(self.npy_path+'/test_image_name.txt','w')
        imgdatas=[]
        imgs=glob.glob(self.test_path+"/*"+self.img_type)#get a list of images' full names(including path)
        for imgname in imgs:
            midname=imgname[imgname.rindex("/")+1:-4]#get image name
            img,img_h,img_w=util.get_image(self.test_path+"/"+midname+self.img_type)
            f.write(midname+self.img_type+" "+str(img_h)+" "+str(img_w)+'\n')
#            img=img_to_array(load_img(self.test_path+"/"+midname))
            imgdatas.append(img)
            if i%100 ==0:
                print('Done:{0}/{1} images'.format(i,len(imgs)))
            i+=1
        imgdatas=np.array(imgdatas,dtype=np.uint8)
        print("loading done")
        np.save(self.npy_path+'/imgs_test.npy',imgdatas)
        print('Saving to npy files done.')
        f.close
        
    def create_val_data(self):
        i=0
        imgdatas=[]
        imglabels=[]
        imgs=glob.glob(self.val_path+"/*"+self.img_type)#get a list of images' full names(including path)
        for imgname in imgs:
            midname=imgname[imgname.rindex("/")+1:-4]#get image name
#            img=img_to_array(load_img(self.data_path+"/"+midname))
            img,img_h,img_w=util.get_image(self.val_path+"/"+midname+self.img_type) 
#            label=img_to_array(load_img(self.label_path+"/"+midname))
            label,y_h,y_w=util.get_label(self.val_label_path+"/"+midname+self.annot_img_type)
            imgdatas.append(img)
            imglabels.append(label)
            if i%100 ==0:
                print('Done:{0}/{1} images'.format(i,len(imgs)))
            i+=1
        imgdatas=np.array(imgdatas,dtype=np.uint8)
        imglabels=np.array(imglabels,dtype=np.uint8)
        print("loading done")
        np.save(self.npy_path+'/imgs_val.npy',imgdatas)
        np.save(self.npy_path+'/imgs_mask_val.npy',imglabels)
        print('Saving to npy files done.')
    
    def load_train_data(self):
        imgs_train=np.load(self.npy_path+"/imgs_train.npy")
        imgs_mask_train=np.load(self.npy_path+"/imgs_mask_train.npy")
        imgs_train=imgs_train.astype('float32')
#        imgs_mask_train=imgs_mask_train.astype('float32')
        imgs_train/=255     
        return imgs_train, imgs_mask_train
    
    def load_labeltest_data(self):
        imgs_test=np.load(self.npy_path+"/imgs_labeltest.npy")
        imgs_mask_test=np.load(self.npy_path+"/imgs_mask_test.npy")
        imgs_test=imgs_test.astype('float32')
#        imgs_mask_test=imgs_mask_test.astype('float32')
        imgs_test/=255     
        return imgs_test, imgs_mask_test
    
    def load_test_data(self):
        imgs_test=np.load(self.npy_path+"/imgs_test.npy")
        imgs_test=imgs_test.astype('float32')
        imgs_test/=255
        return imgs_test
    
    def load_val_data(self):
        imgs_val=np.load(self.npy_path+"/imgs_val.npy")
        imgs_mask_val=np.load(self.npy_path+"/imgs_mask_val.npy")
        imgs_val=imgs_val.astype('float32')
#        imgs_mask_val=imgs_mask_val.astype('float32')
        imgs_val/=255     
        return imgs_val, imgs_mask_val
    
if __name__=="__main__":
    mydata=dataProcess()
#    mydata.create_train_data()
#    mydata.create_test_data()
#    mydata.create_labeltest_data()
#    mydata.create_val_data()
    
    
#    mydata.create_train_edge()
#    mydata.create_test_edge()
#    mydata.create_val_edge()