# 有时候需要把模型保存起来，有时候需要做一些checkpoint 在训练中
# 以至于如果计算机宕机，我们还可以从之前checkpoint的位置去继续
# TensorFlow 使得我们去保存和加载模型非常方便，在构建阶段最后仅需要去创建Saver节点，就可以保存我们checkpoint的位置
# 然后在计算阶段去调用save()方法

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
__author__ = 'check'
# mn.SOURCE_URL = "http://yann.lecun.com/exdb/mnist"
my_nmist = input_data.read_data_sets("MNIST_data_bak/",one_hot=True)

x = tf.placeholder(dtype=tf.float32,shape=(None,784))
# 初始化都是0，二维矩阵784乘以10个W值
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))# b 是截距
# tf.nn.softmax 就是归一化公式 e^z1/e^z1+e^z2+e^z3....+e^zn (y 是归一化的结果即预测的y，也就是y_hat)
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(dtype=tf.float32,shape=(None,10))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
# 0.5 就是学习率
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
# 创建Saver()节点
saver = tf.train.Saver()
n_epoch = 1000

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        if epoch % 100 == 0:
            save_path = saver.save(sess,"./ckpt/my_models.ckpt")
        
        batch_xs,batch_ys = my_nmist.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch_xs,y_: batch_ys})

    best_theta = W.eval()
    save_path = saver.save(sess,"./ckpt/my_model_final.ckpt")