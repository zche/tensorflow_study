import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data_bak/',one_hot=True)
sess = tf.InteractiveSession()

# 截断的正太分布噪声，标准差设为0.1
# 同时因为我们使用Relu，也给偏置项增加一些小的正值0.1用来避免死亡节点(dead neurons)
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷积层和池化层也是接下来要重复使用的，因此也为它们定义创建函数
# tf.nn.conv2d是TensorFlow中的2维卷积函数，参数中X是输入，W是卷积的参数，比如[5,5,1,1]
# 前面两个数字代表卷积核的尺寸，第三个数字代表有多少个channel，因为我们只有灰度单色，所以
# 最后代表核的数量，也就是这个卷积层会提取多少类的特征

# Strides代表卷积模板移动的步长，都是1代表会不遗漏地划过图片的每一个点！Padding代表边界的
# 边界加上Padding让卷积的输出和输入保持同样SAME的尺寸
def conv2d(X,W):
    return tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')

# tf.nn.max_pool是TensorFlow中的最大池化函数，我们这里使用2*2的最大池化，即将2*2的像素均值
# 最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著的特征，因为希望整体上缩小
# ksize(即 kernel size,这里是2x2,相当于把4个点变成1个点)
# strides也设为横竖两个方向以2为步长。如果步长还是1，name我们会得到一个尺寸不变的图片
def max_pool_2x2(X):
    return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 因为卷积神经网络会利用到空间结构信息，因此需要将1D的输入向量转为2D的图片结果，即从1*784
# 同时以为只有一个颜色通道，顾最终尺寸为[-1,28,28,1],前面的-1代表样本数量不固定，最后
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
# -1 代表不确定，1代表一个通道channel
x_image = tf.reshape(x,[-1,28,28,1])

# 定义我的第一个卷积层，我们先使用前面写好的函数进行参数初始化，包括weights 和bias,这里的
# 核尺寸为5*5,1个颜色通道，32个不同的额卷积核，然后使用conv2d函数进行卷积操作，并加上偏置
# 非线性处理，最后，使用最大池化函数max_pool_2*2对卷积的输出结果进行池化操作
# 卷积层中的W就是卷积核，一个卷积核(卷积核就是filter)对应一个FeatureMap（就是卷积层的输出）
# 卷积核就是W，至于卷积核里是什么数值，即W是多少，是计算出来的，需要指定的是卷积核的大小，即几乘以几是需要实现指定的
# 计算卷积核即W是通过梯度下来，先随机以值，然后依次循环计算
# (5,5,1,32) 中的1代表维度 代表通道channels，这里我觉得是上一个输出的维度，即当前卷积层的输入数据的维度
W_conv1 = weight_variable([5,5,1,32]) 
b_conv1 = bias_variable([32])
# relu 做非线性变化
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
#  relu之后就开始做池化
h_pool1 = max_pool_2x2(h_conv1)


# 第二层和第一个一样，但是卷积核变成了64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 因为前面经历了两次步长为2*2的最大池化，所以边长已经只有1/4了，图片尺寸由28*28变成了7*7
# 而第二个卷积层的卷积核数量为64，其输出的tensor尺寸即为7*7*64
# 我们使用tf.reshape函数对第二个卷积层的输出tensor进行变形，将其转成1D的向量
# 然后练级一个全连接层，隐含节点1024，并使用Relu激活函数
# FeatureMap 是64个，池化后也是64个
# [7*7*64,1024] 这里的64 节点是池化后的数量 ，全连接接的就是池化的数量
# 1024代表全连接层有1024个神经元（即隐藏节点个数）
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# 防止过拟合，使用Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# 接 Softmax分类
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 训练
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
        print("step %d,training accuracy %g" % (i,train_accuracy))
    train_step.run(feed_dict={x: batch[0],y_: batch[1],keep_prob: 0.5})
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images,y_: mnist.test.labels,keep_prob: 1.0}))

# 最后，这个CNN模型可以得到的准确率为99.2%,基本可以满足手写数字识别准确率的要求
# 相比之前的MLP的2%的错误率，CNN的错误率下降了大约60%，这里主要的性能提升都来自于更优秀的网络设计
# 即卷积网络对图像特征的提取和抽象能力，依靠卷积核的权值共享，CNN的参数数量并没有爆炸，降低计算量的同时
# 也减轻了过拟合，因此整个模型的性能有较大的提升，这里我们只是实现了一个简单的卷积神经网络，没有负载的Trick
# 接下来我们实现复杂一点的卷积网络，MNIST数据集已经不适合用来评测其性能
# 我们将使用CIFAR-10数据集进行训练，这也是深度学习可以大幅领先其他模型的一个数据集

