import tensorflow as tf
hello = tf.constant("Hello my name")

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
add = node1+node2

sess = tf.Session()
print(sess.run(hello))

print(sess.run(add,feed_dict={node1:3,node2:5}))