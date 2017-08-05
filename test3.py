#!/usr/bin/env python

#Simple test for deeper net.  see https://github.com/nlintz/TensorFlow-Tutorials/blob/master/04_modern_net.py

#This is multi-classification; ie more than 2 outcomes

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data



def init_weights(shape):
    #what is the "Variable" definition for?
    #answer: see https://www.tensorflow.org/api_docs/python/tf/Variable.  Essentially, 
    #this maintains state across calls to run()
    #eg, can take the value assigned to tf.Variable and then call value.assign(newValue)
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


#more complicated model!
#this takes a bunch of variables. 
# X is the vector of inputs (i think)
# w_h is the first hidden layer
# w_h2 is the second hidden layer
# w_o is output layer
# p_keep_input...not sure what this is? 
# it scales the input by 1/p_keep_input
# p_keep_hidden..not sure what this is? 
# it scales the hidden layers by 1/p_keep_hidden
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout

    ##Dropout is here to 
    # scale X by 1/p_keep_input..not sure 
    # what this purpose is
    X = tf.nn.dropout(X, p_keep_input)
    #Max(input, 0)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

#Again, read the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


#Define some placeholder data.  
# Note that this represents a single image
X = tf.placeholder("float", [None, 784])
#and this represents a single output
Y = tf.placeholder("float", [None, 10])

# Initial weights 
# note that the 625=25*25..so it looks like we have 25 nodes per hidden layer?
#Or is this actually the nubmer of nodes!!!???  625 feels like a lot!
w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

#again...placholder for the weird 
# scaling thing in the model above
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

#declare the model...but nothing has been
#initialized yet
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

##Good old softmax on the classifications with cross entropy.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

#Hmm this is a different optimizers.  wht not SGD or GD?
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

#to create prediction of output from the predicted probabilities
predict_op = tf.argmax(py_x, 1)


#And time to actually create the session!
with tf.Session() as sess:
    # initialize all variables...
    tf.global_variables_initializer().run()

    #IS this actually a reasonable number? I guess we will see
    # Is the p_keep_input intended as scaling factor?
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, 
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))