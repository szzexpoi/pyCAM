import sys
sys.path.append('/home/eric/caffe/python/')
import caffe
import numpy as np
from util import soft_attention, load_data, save_map, prepro_img
caffe.set_mode_gpu()
caffe.set_device(0)

def CAM():
    #defining network
    net_weights = './model/imagenet_googleletCAM_train_iter_120000.caffemodel'
    net_model = './model/deploy_googlenetCAM.prototxt'
    net = caffe.Net(net_model, net_weights, caffe.TEST)

    weights_LR = net.params['CAM_fc'][0].data #extracting weights from the last fc layer for weighted sum

    data_path = '/media/eric/New Volume1/Vision/images_sub100/val'
    save_path = './soft_attention'
    data, target, flname = load_data(data_path)
    for i, cur_img in enumerate(data):
        process_img = prepro_img(cur_img)
        net.blobs['data'].data[...] = process_img
        output = net.forward()
        output_prob = output['prob'][0]
        class_idx = np.argmax(output_prob) # selecting the most confidence class to compute soft attention map
        activation_map = net.blobs['CAM_conv'].data[0]
        attention_map = soft_attention(activation_map, weights_LR, class_idx)
        save_map(attention_map, save_path, target[i], flname[i])
        if i%5000 == 0:
            print 'Processed %d out of %d images' %(i,len(data))

if __name__ == "__main__":
    CAM()
