import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from flask import Flask

#그래프
data = pd.read_csv("data2x.csv", names = ['ACI', 'Business_effect', 'Function_error_range', 'Recognition-resolution (m)', 'grade'])
print(data)
data.head()
data.info()
sns.set()
#sns.pairplot(data, hue='grade')
plt.show()
#그래프끝

#분류
tf.set_random_seed(777)  # for reproducibility
x = np.loadtxt('data2x.csv', delimiter=",", usecols=[0,1,2,3]) #자료중에 특성만 저장
y = np.loadtxt('data2x.csv', delimiter=",",dtype=np.str , usecols=[4]) #종류저장

x_data = x[:,0:]
for i in range(len(y)):
    if y[i] =='1grade':
        y[i] = 0
    if y[i] == '2grade':
        y[i] = 1
    elif y[i] == '3grade':
        y[i] = 2
    elif y[i] == '4grade':
        y[i] = 3
    elif y[i] == '5grade':
        y[i] = 4

y_data = np.array([[y] for y in y])
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.007, random_state=0) #학습,테스트 데이터 나누기

nb_classes = 5

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot사용하면 한차원이 더늘어남
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) #원하는 모양으로 shape변경
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# softmax
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot) #softmax_cross_entropy_with_logits사용시 logits를 넣어준다 hypo가 아니다
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(11):
        sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
        if step % 200 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: X_train, Y: y_train})
            print("Step: {:5}\tLoss: {:.3f}\tAccuracy: {:.2%}".format(step, loss, acc))
            #print(acc)

    # Let's see if we can predict
    pred, ac = sess.run([prediction, accuracy], feed_dict={X: X_test, Y:y_test})
    print(ac)
    #y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_test.flatten()):
        print("[{}] Prediction: {} Real Y: {}".format(p == int(y), p, int(y)))

app = Flask(__name__)

@app.route("/")
def hello():
    return 'hello!'

if __name__ == '__main__':
    app.run()