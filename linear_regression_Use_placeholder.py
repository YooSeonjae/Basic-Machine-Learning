import tensorflow as tf

#그래프 구현
#x, y 데이터
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypo = x * w + b

cost = tf.reduce_mean(tf.square(hypo - y))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = opt.minimize(cost)
#그래프 구현 끝

#세션 만들기
sess = tf.Session()
sess.run(tf.global_variables_initializer())#Variable를 사용할 때는 global_variables_initializer를 실행

for step in range(2001):
    cost_val,w_val,b_val, _ = sess.run([cost,w,b,train],feed_dict={x:[1,2,3,4,5],y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 20==0:
        print(step, cost_val, w_val, b_val)

print(sess.run(hypo,feed_dict={x:8}))
