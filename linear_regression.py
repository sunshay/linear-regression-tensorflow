# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:42:59 2017

@author: USER
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

W=np.random.uniform(size=[20,4],low=-1,high=1)
B=np.random.uniform(size=[1,4],low=-1,high=1)
x_data=np.random.uniform(size=[ND,20],low=-1,high=1)
y_data=np.matmul(x_data,W)+B+0.01*np.random.uniform(size=[ND,4],low=-1,high=1)
Ntest=np.int32(ND*0.2)
x_train=x_data[:Ntest]
y_train=y_data[0:Ntest]
x_test=x_data[Ntest:]
y_test=y_data[Ntest:]

x=tf.placeholder(shape=[None,20],dtype=tf.float32)
We=tf.Variable(tf.random_uniform(shape=[20,4]))
Be=tf.Variable(tf.random_uniform(shape=[1,4]))
y=tf.add(tf.matmul(x,We),Be)
yr=tf.placeholder(shape=[None,4],dtype=tf.float32)
loss=tf.reduce_mean(tf.square(y-yr))
optimizer=tf.train.GradientDescentOptimizer(0.1)
train=optimizer.minimize(loss)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
error_train=[]
error_test=[]
j=0
for i in range(1000):
    _=sess.run(train,feed_dict={x:x_train,yr:y_train})
    if i%10==0:
        error=sess.run(loss,feed_dict={x:x_train,yr:y_train})
        error_train.append(error)
        error=sess.run(loss,feed_dict={x:x_test,yr:y_test})
        error_test.append(error)
        #
#
plt.plot(np.arange(0,len(error_train)),np.array(error_train),'b',np.arange(0,len(error_test)),np.array(error_test),'r')
plt.grid('on')
plt.show()

