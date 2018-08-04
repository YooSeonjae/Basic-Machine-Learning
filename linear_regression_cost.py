import tensorflow as tf
import matplotlib.pyplot as plt

x=[1,2,3]
y=[1,2,3]

w = tf.placeholder(tf.float32)

hypo = x*w

cost = tf.reduce_mean(tf.square(hypo-y))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

w_val = []
cost_val = []

for i in range(-30,50):
    feed_w = i*0.1
    curr_cost, curr_w = sess.run([cost,w],feed_dict={w:feed_w})
    w_val.append(curr_w)
    cost_val.append(curr_cost)

plt.plot(w_val,cost_val)
plt.show()

#최소화
#learning_rate = 0.1
#gradient = tf.reduce_mean((w*x-y)*x)
#descent = w - learning_rate*gradient
#update = w.assign(descent)

# Lab 3 Minimizing Cost
tf.set_random_seed(777)  # for reproducibility

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.0)

# Linear model
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
