import tensorflow as tf
#任何创建的节点都会自动加入到默认的图
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph)
graph = tf.Graph()
x3 = tf.Variable(3)
with graph.as_default():
    x2 = tf.Variable(2)
x4 = tf.Variable(4)
print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())
print(x3.graph is tf.get_default_graph())
print(x4.graph is tf.get_default_graph())