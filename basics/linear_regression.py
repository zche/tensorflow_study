import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(data_home="/Users/zhaoruifei/Study/ML/tensorflow/scikit_learn_data",download_if_missing=True)
m , n = housing.data.shape
print(m,n)
print(housing.data)
# 这里添加一个额外的bias输入特征（x0=1）到所有的训练数据上面，因为使用的numpy所有会立即执行
housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]
# 创建两个TensorFlow 常量节点X和y，去持有数据和标签
X = tf.constant(housing_data_plus_bias,dtype=tf.float32,name='x')
y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
#使用一些TensorFlow框架提供的矩阵操作去求theta
XT = tf.transpose(X)
#解析解异步计算出最优解
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)
with tf.Session() as sess:
    theta_value = theta.eval() #sess.run(theta)
    print(theta_value)