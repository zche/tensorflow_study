import tensorflow as tf

with tf.device('/cpu:0'):
    x = tf.Variable(3,name='x')
y = tf.Variable(4,name='y')
f = x*x*y + y +2

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()