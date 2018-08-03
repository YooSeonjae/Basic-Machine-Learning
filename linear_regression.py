import tensorflow as tf

#그래프 구현
#x, y 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypo = x_train * w + b

cost = tf.reduce_mean(tf.square(hypo - y_train))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = opt.minimize(cost)
#그래프 구현 끝

#세션 만들기
sess = tf.Session()
sess.run(tf.global_variables_initializer())#Variable를 사용할 때는 global_variables_initializer를 실행

for step in range(5000):
    sess.run(train)
    if step % 20==0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))
