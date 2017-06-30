import numpy as np
import cv2
from glob import glob
import os

def soft_attention(act_map,weights,class_idx):
    weight = weights[class_idx,:]
    weight = weight.reshape((len(weight),1,1))
    soft_attention = act_map*weight
    soft_attention = np.sum(soft_attention,axis=0)

    return soft_attention

def load_data(path):
    category = glob(path+'/*/')
    data = []
    target = [] #storing the class label for future analysis
    flname = []
    for cur in category:
        imgs = glob(cur+'*.JPEG')
        for i,img in enumerate(imgs):
            cur_name = os.path.basename(img)[:-5]
            I = cv2.imread(img)
            I = cv2.resize(I,(224, 224),interpolation = cv2.INTER_LINEAR)
            data.append(I)
            target.append(cur[len(path)+1:-1])
            flname.append(cur_name)
    return data, target, flname

def save_map(data,path,category,flname):
    norm_map = (data-np.min(data))/(np.max(data)-np.min(data))
    norm_map *= 255
    save_path = os.path.join(path,str(category))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(os.path.join(save_path,flname+'_soft.jpg'),norm_map)

def save_map_salicon(data,path,flname):
    norm_map = (data-np.min(data))/(np.max(data)-np.min(data))
    norm_map *= 255
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(os.path.join(path,flname+'.jpg'),norm_map)

def prepro_img(img):
    mean = [103.939, 116.779, 123.68]
    mean = np.array(mean)
    mean = mean.reshape((1,1,3))
    tmp = np.copy(img)
    tmp = tmp.astype('float32')
    process_img = tmp - mean
    process_img = process_img.reshape((1,process_img.shape[0],process_img.shape[1],process_img.shape[2]))
    process_img = np.tile(process_img,(10,1,1,1))
    process_img = process_img.transpose(0,3,1,2)
    return process_img
