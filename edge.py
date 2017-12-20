import numpy as np
import os
from PIL import Image, ImageFilter

def get_edge(label_path):  
    image = Image.open(label_path)
    image = image.convert("RGB")
    im=image.filter(ImageFilter.CONTOUR)
    #im=image.filter(ImageFilter.FIND_EDGES)#black background
    image = im.convert("P")
    im=np.array(image)
    max_number=np.max(im)
    im[im<max_number]=0
    im[0,:]=max_number
    im[-1,:]=max_number
    im[:,0]=max_number
    im[:,-1]=max_number
    im[im==max_number]=255
    im=Image.fromarray(im)
    return im

if __name__=='__main__':
#    edge_save_dir="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/train_edge"
#    edge_save_dir="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/test_edge"
    edge_save_dir="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/val_edge"

#    file_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/index"   #train
#    file_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/testannot"   #test
    file_path="/home/boriska/Downloads/HELEN/SmithCVPR2013_dataset_resized/valannot"   #val
    file=os.listdir(file_path)#2134243149_2
    for i in range(len(file)):
        label_path=os.path.join(file_path, file[i])
        imglabel=get_edge(label_path)
        imglabel.save(os.path.join(edge_save_dir, file[i]))
        print(".", end="")
