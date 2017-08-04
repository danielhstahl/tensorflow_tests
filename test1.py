#!/usr/bin/env python

#Simple test for tensorflow.  see https://github.com/nlintz/TensorFlow-Tutorials/blob/master/00_multiply.py

import tensorflow as tf

# creates nodes in a graph
# "construction phase"
x1 = tf.constant(5)
x2 = tf.constant(6)

#note this doesn't work too well...I think its an older version of tensorflow
result=tf.multiply(x1, x2)
print(result)

#creates symbolic placeholders
a=tf.placeholder("float")
b=tf.placeholder("float")
y=tf.multiply(a, b)# Note that y is now a function...of the variables a, b yet to be named

with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
    print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))