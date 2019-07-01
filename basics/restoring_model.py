from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
__author__ = 'check'
# mn.SOURCE_URL = "http://yann.lecun.com/exdb/mnist"
my_nmist = input_data.read_data_sets("MNIST_data_bak/",one_hot=True)

x = tf.placeholder(dtype=tf.float32,shape=(None,784))
# 初始化都是0，二维矩阵784乘以10个W值
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))# b 是截距
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(dtype=tf.float32,shape=(None,10))

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"./ckpt/my_model_final.ckpt")
    # 评估
    # tf.argmax() 是一个从tensor中寻找最大值的序号，tf.argmax 就是各个预测的数字中概率最大的那个
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

    # 用tf.cast 将之前correnct_prediction 输出的bool值转换为float32，再求平均
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #accuracy 表示所有的列向量加和平均后的正确率

    # 测试
    print(accuracy.eval({x: my_nmist.test.images, y_: my_nmist.test.labels}))