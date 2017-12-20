import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input  
from keras.utils.np_utils import to_categorical
from skimage.transform import rescale
from keras.preprocessing.image import *
import os


# Pascal VOC color palette for labels
#_PALETTE = [
#        0, 0, 0,
#           128, 0, 0,
#           0, 128, 0,
#           128, 128, 0,
#           0, 0, 128,
#           128, 0, 128,
#           0, 128, 128,
#           128, 128, 128,
#           64, 0, 0,
#           192, 0, 0,
#           64, 128, 0,
#           192, 128, 0,
#           64, 0, 128,
#           192, 0, 128,
#           64, 128, 128,
#           192, 128, 128,
#           0, 64, 0,
#           128, 64, 0,
#           0, 192, 0,
#           128, 192, 0,
#           0, 64, 128,
#           128, 64, 128,
#           0, 192, 128,
#           128, 192, 128,
#           64, 64, 0,
#           192, 64, 0,
#           64, 192, 0,
#           192, 192, 0]
#
_PALETTE = [0, 0, 0,#background
           255, 255, 0,#face skin(excluding ears and neck)
           139, 76, 57,#left eyebrow
           139, 139, 139,#right eyebrow(in picture)
           0, 205, 0,#left eye
           0, 138, 0,#right eye
           154, 50, 205,#nose
           0, 0, 139,#upper lip
           255, 165, 0,#inner lip
           72, 118, 255,#lower lip
           255, 0, 0]#hair
num_classes=11
scale=1./4

def get_label_image(probs, img_h, img_w):
    """ Returns the label image (PNG with Pascal VOC colormap) given the probabilities.

    Note: This method assumes 'channels_last' data format.
    """

    labels = probs.argmax(axis=2).astype("uint8")[:img_h, :img_w]
    label_im = Image.fromarray(labels, "P")
    label_im.putpalette(_PALETTE)
    return label_im    

def get_sig_image(probs, img_h, img_w):
    """ Returns the label image (PNG with Pascal VOC colormap) given the probabilities.

    Note: This method assumes 'channels_last' data format.
    """

    labels = 255*probs[:img_h, :img_w]
#    label_im = Image.fromarray(uint8(labels))
    return labels
     
def get_eNet_label_image(probs, img_h, img_w):
    """ Returns the label image (PNG with Pascal VOC colormap) given the probabilities.

    Note: This method assumes 'channels_last' data format.
    """
    labels = probs.argmax(axis=1).astype("uint8")
    labels=probs.reshape(512,512,num_classes)[:img_h, :img_w]
    label_im = Image.fromarray(labels, "P")
    label_im.putpalette(_PALETTE)
    return label_im

def get_edge(file_name):    
    #Note: channels last  
    im=Image.open(file_name)
    img_w,img_h=im.size
    pad_h=500-img_h
    pad_w=500-img_w  
    if pad_h<0 or pad_w<0:
        x_lu=0
        y_lu=0
        x_rl=img_w
        y_rl=img_h
        if pad_h<0:
            y_lu=int(abs(pad_h)/2)
            y_rl=img_h-abs(pad_h)+y_lu
        if pad_w<0:
            x_lu=int(abs(pad_w)/2)
            x_rl=img_w-abs(pad_w)+x_lu
        im=im.crop((x_lu,y_lu,x_rl,y_rl))
        pad_h=0
        pad_w=0
    im=np.array(im)  
    img_h,img_w=im.shape  
    if img_h>500 or img_w>500:  
        raise ValueError("Please resize your images to be not bigger than 500 x 500")  
    im=np.pad(im,pad_width=((0,pad_h),(0,pad_w)),mode='constant',constant_values=255)  
    return im.astype(np.uint8).reshape(500,500,1),img_h,img_w  

def get_label(file_name):    
    #Note: channels last  
    im=Image.open(file_name)
    img_w,img_h=im.size
    pad_h=500-img_h
    pad_w=500-img_w  
    if pad_h<0 or pad_w<0:
        x_lu=0
        y_lu=0
        x_rl=img_w
        y_rl=img_h
        if pad_h<0:
            y_lu=int(abs(pad_h)/2)
            y_rl=img_h-abs(pad_h)+y_lu
        if pad_w<0:
            x_lu=int(abs(pad_w)/2)
            x_rl=img_w-abs(pad_w)+x_lu
        im=im.crop((x_lu,y_lu,x_rl,y_rl))
        pad_h=0
        pad_w=0
    im=np.array(im)  
    img_h,img_w=im.shape  
    im=to_categorical(im,num_classes)
    im=im.reshape(img_h,img_w,num_classes)
    if img_h>500 or img_w>500:  
        raise ValueError("Please resize your images to be not bigger than 500 x 500")  
    im=np.pad(im,pad_width=((0,pad_h),(0,pad_w),(0,0)),mode='constant',constant_values=0)  
    return im.astype(np.uint8).reshape(500,500,num_classes),img_h,img_w  

def get_test_label(file_name):    
    #Note: channels last  
    im=Image.open(file_name)
    img_w,img_h=im.size
    pad_h=500-img_h
    pad_w=500-img_w  
    if pad_h<0 or pad_w<0:
        x_lu=0
        y_lu=0
        x_rl=img_w
        y_rl=img_h
        if pad_h<0:
            y_lu=int(abs(pad_h)/2)
            y_rl=img_h-abs(pad_h)+y_lu
        if pad_w<0:
            x_lu=int(abs(pad_w)/2)
            x_rl=img_w-abs(pad_w)+x_lu
        im=im.crop((x_lu,y_lu,x_rl,y_rl))
        pad_h=0
        pad_w=0
    im=np.array(im)  
    img_h,img_w=im.shape  
    if img_h>500 or img_w>500:  
        raise ValueError("Please resize your images to be not bigger than 500 x 500")  
    im=np.pad(im,pad_width=((0,pad_h),(0,pad_w)),mode='constant',constant_values=0)  
    return im.astype(np.uint8).reshape(500,500),img_h,img_w  
      
def get_image(file_name):  
        #Note: channels last  
    im=Image.open(file_name)
    img_w,img_h=im.size
    pad_h=500-img_h
    pad_w=500-img_w  
    if pad_h<0 or pad_w<0:
        x_lu=0
        y_lu=0
        x_rl=img_w
        y_rl=img_h
        if pad_h<0:
            y_lu=int(abs(pad_h)/2)
            y_rl=img_h-abs(pad_h)+y_lu
            pad_h=0
        if pad_w<0:
            x_lu=int(abs(pad_w)/2)
            x_rl=img_w-abs(pad_w)+x_lu
            pad_w=0
        im=im.crop((x_lu,y_lu,x_rl,y_rl))
    im=np.array(im).astype(np.float32)
    assert im.ndim==3,"Only RGB images are supported"  
    img_h,img_w,img_c=im.shape  
    assert img_c==3,"Only RGB images are supported"  
    if img_h>500 or img_w>500:  
        raise ValueError("Please resize your images to be not bigger than 500 x 500")  
    im=preprocess_input(im)  
    im=np.pad(im,pad_width=((0,pad_h),(0,pad_w),(0,0)),mode='constant',constant_values=0)  
    return im.astype(np.uint8).reshape(500,500,3),img_h,img_w  

def get_eNet_image(file_name):  
        #Note: channels last  
    im=Image.open(file_name)
    img_w,img_h=im.size
    pad_h=512-img_h
    pad_w=512-img_w  
    if pad_h<0 or pad_w<0:
        x_lu=0
        y_lu=0
        x_rl=img_w
        y_rl=img_h
        if pad_h<0:
            y_lu=int(abs(pad_h)/2)
            y_rl=img_h-abs(pad_h)+y_lu
            pad_h=0
        if pad_w<0:
            x_lu=int(abs(pad_w)/2)
            x_rl=img_w-abs(pad_w)+x_lu
            pad_w=0
        im=im.crop((x_lu,y_lu,x_rl,y_rl))
    im=np.array(im).astype(np.float32)
    assert im.ndim==3,"Only RGB images are supported"  
    img_h,img_w,img_c=im.shape  
    assert img_c==3,"Only RGB images are supported"  
    if img_h>512 or img_w>512:  
        raise ValueError("Please resize your images to be not bigger than 512 x 512")  
    im=preprocess_input(im)  
    im=np.pad(im,pad_width=((0,pad_h),(0,pad_w),(0,0)),mode='constant',constant_values=0)  
    return im.astype(np.uint8).reshape(512,512,3),img_h,img_w 

def get_eNet_label(file_name):    
    #Note: channels last  
    im=Image.open(file_name)
    img_w,img_h=im.size
    pad_h=512-img_h
    pad_w=512-img_w  
    if pad_h<0 or pad_w<0:
        x_lu=0
        y_lu=0
        x_rl=img_w
        y_rl=img_h
        if pad_h<0:
            y_lu=int(abs(pad_h)/2)
            y_rl=img_h-abs(pad_h)+y_lu
        if pad_w<0:
            x_lu=int(abs(pad_w)/2)
            x_rl=img_w-abs(pad_w)+x_lu
        im=im.crop((x_lu,y_lu,x_rl,y_rl))
        pad_h=0
        pad_w=0
    im=np.array(im)  
    img_h,img_w=im.shape  
    im=to_categorical(im,num_classes)
    im=im.reshape(img_h,img_w,num_classes)
    if img_h>512 or img_w>512:  
        raise ValueError("Please resize your images to be not bigger than 512 x 512")  
    im=np.pad(im,pad_width=((0,pad_h),(0,pad_w),(0,0)),mode='constant',constant_values=0)  
    return im.astype(np.uint8).reshape(512*512,num_classes),img_h,img_w  
      
def get_resized_image(file_name):  
        #Note: channels last  
    im=np.array(Image.open(file_name)).astype(np.float32)  
    assert im.ndim==3,"Only RGB images are supported"  
    im=preprocess_input(im)/255
    im=rescale(im,scale)
    im*=255
    img_h,img_w,img_c=im.shape
    pad_h=150-img_h 
    pad_w=150-img_w  
    im=np.pad(im,pad_width=((0,pad_h),(0,pad_w),(0,0)),mode='constant',constant_values=0)  
    return im.astype(np.uint8).reshape(150,150,3),img_h,img_w 

def get_resized_label(file_name):    
    #Note: channels last  
    im=np.array(Image.open(file_name))  
    im=rescale(im,scale)
    img_h,img_w=im.shape  
    im=to_categorical(im,num_classes)
    im=im.reshape(img_h,img_w,num_classes) 
    pad_h=150-img_h  
    pad_w=150-img_w  
    im=np.pad(im,pad_width=((0,pad_h),(0,pad_w),(0,0)),mode='constant',constant_values=0)  
    return im.astype(np.uint8).reshape(150,150,num_classes),img_h,img_w  

class SegDirectoryIterator(Iterator):
    '''
    label images should be png, where pixel values represent class number
    find images -name *.jpg>image.txt
    find images -name *.png>labels.txt
    
    file_path: location of train.txt or val.txt
    label_suffix: .png or .npy
    loss_shape: shape to use when applying loss funtion to the label data
    '''
    def __init__(self, file_path, seg_data_generator, data_dir ,data_suffix, 
                 label_dir, label_suffix, classes, ignore_label=255, 
                 crop_mode='none', label_cval=255, pad_size=None,
                 target_size=None, color_mode='rgb', class_mode='sparse', #color_mode=rgb or grayscale, class_mode=None or sparse
                 batch_size=1, shuffle=True, seed=None, save_to_dir=None,
                 save_prefix='', save_format='png', loss_shape=None):
        self.file_path=file_path
        self.data_dir=data_dir
        self.data_suffix=data_suffix
        self.label_suffix=label_suffix
        self.label_dir=label_dir
        self.classes=classes
        self.seg_data_generator=seg_data_generator
        self.target_size=tuple(target_size)
        self.ignore_label=ignore_label
        self.crop_mode=crop_mode
        self.label_cval=label_cval
        self.pad_size=pad_size
        self.color_mode=color_mode
        self.nb_label_ch=1
        self.loss_shape=loss_shape
        self.class_mode=class_mode
        self.save_to_dir=save_to_dir
        self.save_prefix=save_prefix
        self.save_format=save_format
        
        if (self.label_suffix=='.npy') or (self.label_suffix=='npy'):
            self.label_file_format='npy'
        else:
            self.label_file_format='img'
            
        if target_size:
            if self.color_mode=='rgb':
                self.image_shape=self.target_size+(3,)
            else:
                self.image_shape=self.target_size+(1,)
            self.label_shape=self.target_size+(self.nb_label_ch,)
        elif batch_size!=1:
            raise ValueError('Batch size must be 1 when target size is undetermined')
        else:
            self.image_shape=None
            self.label_shape=None
        
        if save_to_dir:
            self.palette=None
        
        white_list_formats={'png', 'jpg', 'jpeg', 'bmp', 'npy'}
        self.data_files=[]
        self.label_files=[]
        fp=open(file_path)
        lines=fp.readlines()
        fp.close()
        self.np_sample=len(lines)
        for line in lines:
            line=line.strip('\n')
            self.data_files.append(line+ data_suffix)
            self.label_files.append(line+ label_suffix)
        super(SegDirectoryIterator, self).__init__(self.nb_sample, batch_size, 
                                                   shuffle, seed)
    
    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size=next(self.index_generator)
        if self.target_size:
            batch_x=np.zeros((current_batch_size,)+self.image_shape)
            if self.loss_shape is None and self.label_file_format is 'img':
                batch_y=np.zeros((current_batch_size,)+ self.label_shape, dtype=int)
            elif self.loss_shape is None:
                batch_y=np.zeros((current_batch_size,)+ self.label_shape)
            else:
                batch_y=np.zeros((current_batch_size,)+ self.loss_shape, dtype=np.uint8)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data and labels
        for i, j in enumerate(index_array):
            data_file = self.data_files[j]
            label_file = self.label_files[j]
            img_file_format = 'img'
            img = load_img(os.path.join(self.data_dir, data_file),
                           grayscale=grayscale, target_size=None)
            label_filepath = os.path.join(self.label_dir, label_file)

            if self.label_file_format == 'npy':
                y = np.load(label_filepath)
            else:
                label = Image.open(label_filepath)
                if self.save_to_dir and self.palette is None:
                    self.palette = label.palette

            # do padding
            if self.target_size:
                if self.crop_mode != 'none':
                    x = img_to_array(img, data_format=self.data_format)
                    if self.label_file_format is not 'npy':
                        y = img_to_array(
                            label, data_format=self.data_format).astype(int)
                    img_w, img_h = img.size
                    if self.pad_size:
                        pad_w = max(self.pad_size[1] - img_w, 0)
                        pad_h = max(self.pad_size[0] - img_h, 0)
                    else:
                        pad_w = max(self.target_size[1] - img_w, 0)
                    x = np.lib.pad(x, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2), (0, 0)), 'constant', constant_values=0.)
                    y = np.lib.pad(y, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2), (0, 0)), 'constant', constant_values=self.label_cval)
                else:
                    x = img_to_array(img.resize((self.target_size[1], self.target_size[0]),
                                                Image.BILINEAR),
                                     data_format=self.data_format)
                    if self.label_file_format is not 'npy':
                        y = img_to_array(label.resize((self.target_size[1], self.target_size[
                                         0]), Image.NEAREST), data_format=self.data_format).astype(int)
                    else:
                        print('ERROR: resize not implemented for label npy file')

            if self.target_size is None:
                batch_x = np.zeros((current_batch_size,) + x.shape)
                if self.loss_shape is not None:
                    batch_y = np.zeros((current_batch_size,) + self.loss_shape)
                else:
                    batch_y = np.zeros((current_batch_size,) + y.shape)

            x, y = self.seg_data_generator.random_transform(x, y)
            x = self.seg_data_generator.standardize(x)

            if self.ignore_label:
                y[np.where(y == self.ignore_label)] = self.classes

            if self.loss_shape is not None:
                y = np.reshape(y, self.loss_shape)

            batch_x[i] = x
            batch_y[i] = y
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                label = batch_y[i][:, :, 0].astype('uint8')
                label[np.where(label == self.classes)] = self.ignore_label
                label = Image.fromarray(label, mode='P')
                label.palette = self.palette
                fname = '{prefix}_{index}_{hash}'.format(prefix=self.save_prefix,
                                                         index=current_index + i,
                                                         hash=np.random.randint(1e4))
                img.save(os.path.join(self.save_to_dir, 'img_' +
                                      fname + '.{format}'.format(format=self.save_format)))
                label.save(os.path.join(self.save_to_dir,
                                        'label_' + fname + '.png'))
        # return
        batch_x = preprocess_input(batch_x)
        if self.class_mode == 'sparse':
            return batch_x, batch_y
        else:
            return batch_x
   
        

class SegDataGenerator(object):
    def __int__(self, featurewise_center=False, samplewise_center=False, 
                featurewise_std_normalization=False, 
                samplewise_std_normalization=False,
                channelwise_center=False, 
                rotation_range=0., width_shift_range=0., height_shift_range=0., 
                shear_range=0., zoom_range=0., zoom_maintain_shape=True, 
                channel_shift_range=0., fill_mode='constant', 
                cval=0, label_cval=255, 
                crop_mode='none', crop_size=(0,0), pad_size=None, #crop_mode="none","random","center"
                horizontal_flip=False, vertical_flip=False, rescale=None):
        self.__dict__.update(locals())
        self.mean=None
        self.ch_mean=None
        self.std=None
        self.principal_components=None
        self.rescale=rescale
        self.channel_index=3
        self.row_index=1
        self.col_index=2
        if np.isscalar(zoom_range):
            self.zoom_range=[1-zoom_range, 1+zoom_range]
        elif len(zoom_range)==2:
            self.zoom_range=[zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float of a tuple or list of two floats.'
                            'Received arg:', zoom_range)
        
    def flow_from_directory(self, file_path, data_dir, data_suffix, 
                            label_dir, label_suffix, classes,
                            ignore_label=255, target_size=None, color_mode='rgb',
                            class_mode='sparse', batch_size=32, shuffle=True,
                            seed=None, save_to_dir=None, save_prefix='', 
                            save_format='png', loss_shape=None):
        if self.crop_mode=='random' or self.crop_mode=='center':
            target_size=self.crop_size
        return SegDirectoryIterator(self, file_path=file_path, data_dir=data_dir,
                                    data_suffix=data_suffix, label_dir=label_dir,
                                    label_suffix=label_suffix, classes=classes,
                                    ignore_label=ignore_label, 
                                    crop_mode=self.crop_mode, 
                                    label_cval=self.label_cval,
                                    pad_size=self.pad_size,
                                    target_size=target_size, color_mode=color_mode,
                                    class_mode=class_mode, batch_size=batch_size, 
                                    shuffle=shuffle, seed=seed, 
                                    save_to_dir=save_to_dir, save_prefix=save_prefix,
                                    save_format=save_format, loss_shape=loss_shape)
    