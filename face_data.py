import numpy as np
import os
from PIL import Image


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
_COLOR_LIST=np.array(_PALETTE).reshape(11,3).tolist()
channel=["00","01","02","03","04","05","06","07","08","09","10"]
img_type=".png"
rgb_save_dir="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/rgb"
index_save_dir="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/index"
def get_index(label_path):    
    imglabels=[]
    for i in range(11):
        midname=label_path[label_path.rindex("/")+1:]#get image name
        im=Image.open(label_path+'/'+midname+'_lbl'+channel[i]+img_type)
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
#        img_h,img_w=im.shape
#        if img_h>500 or img_w>500:  
#            raise ValueError("Please resize your images to be not bigger than 500 x 500")
#        im=np.pad(im,pad_width=((0,pad_h),(0,pad_w)),mode='constant',constant_values=0) 
        imglabels.append(im.astype(np.uint8))
    imglabel=np.argmax(np.array(imglabels),axis=0)
    return imglabel

def get_rgb(imglabel,img_h,img_w):
    x=np.zeros([img_h,img_w,3])
    for i in range(img_h):
        for j in range(img_w):
            r,g,b=_COLOR_LIST[int(imglabel[i][j])]
            x[i,j,0]=r
            x[i,j,1]=g
            x[i,j,2]=b
    r=Image.fromarray(x[:,:,0]).convert('L')
    g=Image.fromarray(x[:,:,1]).convert('L')
    b=Image.fromarray(x[:,:,2]).convert('L')
    image=Image.merge("RGB",(r,g,b))
    return image

helen_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/labels"
file=os.listdir(helen_path)#2134243149_2
for i in range(len(file)):
    label_path=os.path.join(helen_path,file[i])
    imglabel=get_index(label_path).astype("uint8")
    label_im = Image.fromarray(imglabel, "P")
    label_im.putpalette(_PALETTE)
    label_im.save(os.path.join(index_save_dir,file[i])+img_type)
    print(".",end="")
            