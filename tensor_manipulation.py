import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape

t = tf.constant([1,2,3,4])
tf.shape(t).eval()

t = tf.constant([[1,2],
                 [3,4]])
tf.shape(t).eval()

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
tf.matmul(matrix1, matrix2).eval()
(matrix1*matrix2).eval()# 완전히 다른값

#브로드캐스팅
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
(matrix1+matrix2).eval()
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1+matrix2).eval()

#Reduce Mean = 평균
tf.reduce_mean([1, 2], axis=0).eval() #정수
x = [[1., 2.],[3., 4.]] #float
tf.reduce_mean(x).eval()
tf.reduce_mean(x, axis=0).eval() # axis 즉 축에 따라 값이 달라짐
tf.reduce_mean(x, axis=1).eval()
tf.reduce_mean(x, axis=-1).eval()

#Reduce sum = 합
tf.reduce_sum(x).eval()
tf.reduce_sum(x, axis=0).eval()
tf.reduce_sum(x, axis=1).eval()
tf.reduce_sum(x, axis=-1).eval()

#Argmax = 큰 값의 자리
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()
tf.argmax(x, axis=1).eval()
tf.argmax(x, axis=-1).eval()

#Reshape = 형태 바꾸기
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
t.shape
tf.reshape(t, shape=[-1, 3]).eval()
tf.reshape(t, shape=[-1, 1, 3]).eval()

#squeeze = 합치기
tf.squeeze([[0], [1], [2]]).eval()

#expand_dims = 차원을 하나 늘려서 나누기 같은건 합치기
tf.expand_dims([0, 1, 2], 1).eval()

#one_hot 제일 큰 자리의 위치만 표시
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval()

#casting
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()

#stack
x = [1, 4]
y = [2, 5]
z = [3, 6]
tf.stack([x, y, z]).eval()
tf.stack([x, y, z], axis=1).eval()

#Ones like and Zeros like
x = [[0, 1, 2],
     [2, 1, 0]]

tf.ones_like(x).eval() #같은 형태지만 값은 다 1으로 만들기
tf.zeros_like(x).eval() #같은 형태지만 값은 다 0으로 만들기

#Zip
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
