import onnx
from onnx_tf.backend import prepare

import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.inf)

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

onnx_model = onnx.load("resnet18v2/resnet18v2.onnx")  # load onnx model
input = np.asarray(Image.open('resnet18v2/man.jpg'))
input = np.asarray([input[:,:,0], input[:,:,1], input[:,:,2]])

input_wrapped = []
input_wrapped.append(input)
input_wrapped = np.asarray(input_wrapped)
prepared_model = prepare(onnx_model)
output = prepared_model.run(preprocess(input_wrapped))  # run the loaded model
print('FLOPS: ', prepared_model.total_float_ops)
print(output[0][0].tolist().index(np.max(output[0][0])))

