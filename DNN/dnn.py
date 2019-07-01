import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 
import numpy as np
from tensorflow.contrib.layers import fully_connected

# 构建图阶段
# 28* 28 = 784 正好是mnist 的每张图片的像素数量（784个维度）
n_inputs = 28 *28
# hidden1有300个隐藏节点
n_hidden1 = 300
n_hidden2 = 100
# 输出层有10个节点(数据集是10分类)
n_outputs = 10

X= tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
y = tf.placeholder(tf.int64,shape=(None),name='y')

'''
# 构建神经网络层，我们这里两个隐藏层，基本一样，除了输入inputs到每个神经元的链接不同
# 和神经元个数不同，输出层也非常相似，只是激活函数从Relu 变成了Softmax而已
def neuron_layer(X,n_neurons,name,activation=None):
    # 包含所有计算节点对于这一层，name_scope 可写可不写
    with tf.name_scope(name):
        # 取输入矩阵的维度作为层的输入连接个数
        n_inputs= int(X.get_shape()[1])
        # 标准方差
        stddev = 2 / np.sqrt(n_inputs)
        # 这里面的w可以看成是二维数组，每个神经元对于一组w参数
        # truncated normal distribution 比 regular normal distribution 的值小
        # 不会出现任何大的权重值，确保慢慢的稳健的训练
        # 使用这种标准方差会让收敛块
        # w参数需要随机，不能为0，否则输出为0，最后调整都是一个幅度没意义
        init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        w = tf.Variable(init,name='weights')
        b = tf.Variable(tf.zeros([n_neurons]),name='biases')
        # 向量表达的使用比一条一条加和要高效
        z = tf.matmul(X,w)+b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z
'''

with tf.name_scope("dnn"):
    # tensorflow 使用这个函数帮助我们使用合适的初始化w和b 的策略，
    # 默认使用Relu 激活函数
    hidden1 = fully_connected(X,n_hidden1,scope="hidden1")
    hidden2 = fully_connected(hidden1,n_hidden2,scope="hidden2")
    # 进入到softmax之前的结果
    logits = fully_connected(hidden2,n_outputs,scope="outputs",activation_fn=None)
    # 输出层的计算分两步
    # 第一步，加和这里就是 logits的结果(就是s1,s2...sn)
    # 第二步 应用softmax函数:一个类别：(e^s1/e^s1+e^s2+...+e^sn) + (e^s2/e^s1+e^s2+...+e^sn) +... (e^sn/e^s1+e^s2+...+e^sn)
    # 10个类别的话，在上面基础上再加和
with tf.name_scope("loss"):
    # 定义交叉熵损失函数，并且求个样本平均
    # 函数等价于先使用softmax损失函数，再接着计算交叉熵，并且更有效率
    # 类似的softmax_cross_entropy_with_logits 只会给one-hot编码，我们使用的会给0-9分类号
    # 如果labels= y 是矩阵 就用softmax_cross_entropy_with_logits 
    # 如果是列向量(分类号)，就用sparse_softmax_cross_entropy_with_logits
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    # reduce_mean  加和求平均
    loss = tf.reduce_mean(xentropy,name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    # 获取logits 里面最大的那1位和y比较类别号是否相同，返回True 或者False 一组值
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 计算图阶段
mnist = input_data.read_data_sets("/Users/zhaoruifei/Study/ML/tensorflow/MNIST_data_bak/")
n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X: X_batch,y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch,y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

        print(epoch,"Train accuracy:",acc_train,"Test accuracy", acc_test)
    save_path = saver.save(sess, "./my_dnn_model_final.ckpt")


    

