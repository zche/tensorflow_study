import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
import matplotlib.pyplot as plt 
import sys

print(sys.path)

# 加载数据集
# 输入图片通常是3D，[hieght,width,channels]
# mini-batch 通常是4D，[mini-batch size,height,width,channels]
dataset = np.array(load_sample_images().images,dtype=np.float32)
# 数据集里面两张图片，一个中国庙宇，一个花
batch_size, height,width,channels = dataset.shape
# 2 427 640 3
print(batch_size,height,width,channels)

# 创建两个filter
# 高，宽，通道，卷积核
# 7,7,channels,2
filters_test = np.zeros(shape=(7,7,channels,2),dtype=np.float32)
filters_test[:,3,:,0] = 1 #垂直
filters_test[3,:,:,1] = 1 #水平

# filter参数是一个filters的集合
X = tf.placeholder(tf.float32,shape=(None,height,width,channels))
# strides(步长) = [1,2,2,1] 中第一最后一个为1，中间对应height和width
# 如果要把卷积后的图片大小保持和上一层的大小一样（same），就必须把卷积后的图片里填充0（Zero Padding）
convolution = tf.nn.conv2d(X,filter=filters_test,strides=[1,2,2,1],padding='SAME')

with tf.Session() as sess:
    output = sess.run(convolution,feed_dict={X: dataset})

plt.imshow(load_sample_images().images[0]) 
plt.show()

# 绘制第一个图的第二个特征图
# 第一个0代表第一张图片，最后一个0代表垂直特征图
plt.imshow(output[0,:,:,0]) 
plt.show()
