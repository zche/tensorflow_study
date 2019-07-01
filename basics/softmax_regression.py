from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
__author__ = 'check'
# mn.SOURCE_URL = "http://yann.lecun.com/exdb/mnist"
my_nmist = input_data.read_data_sets("MNIST_data_bak/",one_hot=True)

# The MNIST data is split into three parts:
# 55000 data points of training data (mnist.train)
# 10000 points of test data (minist.test),amd
# 5000 points of validation data (mnist.validation)

# Each image is 28 pixels by 28 pixels

# 输入的是一堆图片，None 标识不限输入条数，784表示每张图片都是一个784个像素值的一维向量
# 所以输入的矩阵是None 乘以784 二维矩阵
x = tf.placeholder(dtype=tf.float32,shape=(None,784))
# 初始化都是0，二维矩阵784乘以10个W值
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))# b 是截距
# tf.nn.softmax 就是归一化公式 e^z1/e^z1+e^z2+e^z3....+e^zn (y 是归一化的结果即预测的y，也就是y_hat)
y = tf.nn.softmax(tf.matmul(x,W)+b)

# 训练
# lables 是每张图片都对应一个one-hot的10个值的向量
y_ = tf.placeholder(dtype=tf.float32,shape=(None,10))
# 定义损失函数，交叉熵损失函数 (-yi*logPi)m行加和，其中Pi是预测出来归一化的概率值
# 对于多分类问题，通常使用交叉熵损失函数
# reduction_indices 等价于axis，指明按照每行加，还是按照每列加
# y_ 是真实的概率值
# reduce_sum 意思是把里面所有的值加和成一个值
# reduce_mean 加和求平均（默认是行相加求平均，可以通过 reduction_indices 来更改默认行为）
# reduction_indices 意思是：0 代表行， 1 调拨列，这里是按照列进行加和
# y_ * tf.log(y) 是对应位置的向量相乘，如果用matmul 则是行向量 乘以列向量
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
# 0.5 就是学习率
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 评估
# tf.argmax() 是一个从tensor中寻找最大值的序号，tf.argmax就是求各个预测的数字中概率最大的那一个
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

# 用tf.cast 将之前correnct_prediction 输出的bool值转换为float32，再求平均

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #accuracy 表示所有的列向量加和平均后的正确率

# 初始化变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs,batch_ys = my_nmist.train.next_batch(100)# next_batch 从训练集中取100条（mini-batch GD）如果是BGD的话，每次就是全量数据训练而不是一部分
    sess.run(train_step,feed_dict={x: batch_xs,y_: batch_ys})
    print("TrainSet batch acc : %s " % accuracy.eval({x: batch_xs,y_:batch_ys}))
    print("ValidSet acc : %s " % accuracy.eval({x: my_nmist.validation.images, y_: my_nmist.validation.labels}))

# 测试
print("TestSet acc : %s " % accuracy.eval({x: my_nmist.test.images, y_: my_nmist.test.labels}))
sess.close()
# 总结
# 1.定义算法公司，也就是神经网络forward时的计算
# 2. 定义loss，选定优化器，并制定优化器优化loss
# 3.迭代地对数据进行训练
# 4.在测试集或验证集上对准确率进行评测


