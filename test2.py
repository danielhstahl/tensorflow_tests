#!/usr/bin/env python

#Simple test for logisitic regression.  see https://github.com/nlintz/TensorFlow-Tutorials/blob/master/02_logistic_regression.py

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

#Simple linear model, we'll specify the output's sigmoid shape later
def model(X, w):
    return tf.matmul(X, w) 

#reads in data.  Don't really know what one_hot is for
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#assign the sections of the mnist dataset to variables.  
#this is for convenience
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#Assigns names to placeholders
X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

#initial weights are randomly generated
w = init_weights([784, 10]) 

#this is essentially a function.  
#X and w are either placeholders or 
#elements that can be assigned to
py_x=model(X, w)

#The cost function: note the softmax 
# (which acts like a sigmoid for binary classification)
#Interestingly, this includes the cross entropy loss function
#Why is the term "logit" even needed?
#Answer: the logic refers to the input; in this case the py_x logits
#The reduce_mean essentially applies a mean function to the resulting
#vector.  "reduce" is used in the sense of map reduce
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))


##Creates the optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer

#Note that this is a function which can be called later.
#This picks the largest of the 10 "probabilities" for
# classification as the prediction
predict_op = tf.argmax(py_x, 1) 


#now for actually creating stuff!
# always need a session it appears
with tf.Session() as sess:
    # Hmm which variables are initialized here? all? and to what?
    # or is this the equivalent of a lazy load for the functions above?
    tf.global_variables_initializer().run()

    #train 100 times...probably plenty for basic logistic
    for i in range(100):
        ##zip creates tuples from the two provided arrays...ie, turns 
        #zip([1, 2, 3], [4, 5, 6]) into [(1, 4), (2, 5), (3, 6)]
        # range's third argument says how many to skip.  
        # range(0, 6, 2) returns [0, 2, 4]
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        #see how well this is working on test training
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))