import sys
sys.path.append('/home/eric/caffe/python/')
import caffe
import numpy as np
from glob import glob
import cv2
import os
from util import soft_attention, load_data, save_map, prepro_img,save_map_salicon
caffe.set_mode_gpu()
caffe.set_device(0)

def CAM():
    #defining network
    net_weights = './model/imagenet_googleletCAM_train_iter_120000.caffemodel'
    net_model = './model/deploy_googlenetCAM.prototxt'
    net = caffe.Net(net_model, net_weights, caffe.TEST)

    weights_LR = net.params['CAM_fc'][0].data #extracting weights from the last fc layer for weighted sum

    data_path = '/home/eric/Desktop/experiment/salicon/salicon-api/images/val'
    save_path = './soft_attention_salicon'
    file_ = glob(os.path.join(data_path,'*.jpg'))

    for i,cur_img in enumerate(file_):
        cur_data = cv2.imread(cur_img)
        cur_data = cv2.resize(cur_data,(224, 224),interpolation = cv2.INTER_LINEAR)
        process_img = prepro_img(cur_data)
        net.blobs['data'].data[...] = process_img
        output = net.forward()
        output_prob = output['prob'][0]
        class_idx = np.argmax(output_prob) # selecting the most confidence class to compute soft attention map
        activation_map = net.blobs['CAM_conv'].data[0]
        attention_map = soft_attention(activation_map, weights_LR, class_idx)
        save_map_salicon(attention_map, save_path, os.path.basename(cur_img))
        if i%5000 == 0:
            print 'Processed %d out of %d images' %(i,len(file_))

if __name__ == "__main__":
    CAM()
