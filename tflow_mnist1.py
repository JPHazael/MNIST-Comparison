## Imports the MNIST example data 55k training characters, 10k test characters,
## and 5k validation characters. All of the characters have been flattened from 
## 28pixel x 28 pixel images into a 784 dimension vector. Then an array is made out of 
## by multiplying the number of characters by the number of dimensions (784 x 55000)
##for our training data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


## Import tensorflow to do the math and regression

import tensorflow as tf

## Implement our first placeholder for x. This will sit in for x until we actually run a 
##test. I think TF does this to run more efficiently. 
##We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]. 
##Here None means that a dimension can be of any length.


x = tf.placeholder(tf.float32, [None, 784])


##Set tf Variables for the weight and bias. A Variable is a modifiable tensor 
##that lives in TensorFlow's graph of interacting operations. 
##It can be used and even modified by the computation. 
##For machine learning applications, one generally has the model parameters be Variables.
## We set W and b to zeroes initially and then they will gain values as we run our tests.

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

##First, we multiply x(inputs) by W(wieght) with the expression tf.matmul(x, W). 
##This is flipped from when we multiplied them in our equation, where we had Wx, 
##as a small trick to deal with x being a 2D tensor with multiple inputs.
##We then add b(bias toward more likely outcomes), 
##and finally apply tf.nn.softmax. (Softmax converts the weighted inputs into 
##a probability distribution and finally tells us which is the most probable y(label)
##based on the initial x.

y = tf.nn.softmax(tf.matmul(x, W) + b)

## Implement a placeholder variable for y_. We'll talk more about this later.

y_ = tf.placeholder(tf.float32, [None, 10])

##Then we run cross entropy. In information theory, the cross entropy between 
##two probability distributions over the same underlying set of events measures 
##the average amount of information needed to identify an event drawn from the set. 
## The CE method can be applied to static and noisy combinatorial optimization problems.
##In a nutshell the CE method consists of two phases:
##Generate a random data sample (trajectories, vectors, etc.) 
##according to a specified mechanism.
##Update the parameters of the random mechanism based on the data to produce 
##a "better" sample in the next iteration.

##I THINK what this means is that CE takes the information needed to determine 
##the probability of an x(input) being a y(label), our prediction
##and compares it with the info neededto determine whether a randomly generated input
##is a y. 
##By comparing the two CE updates the next iteration of the x to y test 
##to make it more efficient. 

##In our CE process y is our predicted outcome and y_ is our true outcome.
## we set y_ to a placeholder on line 43.

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

##In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent 
##algorithm with a learning rate of 0.01. Gradient descent is a simple procedure, 
##where TensorFlow simply shifts each variable a little bit in the direction that 
##reduces the cost(error rate of our predictions).

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

##gotta initialize the variables before you can run anything

init = tf.initialize_all_variables()

## Now we can begin our tensorflow session 

sess = tf.Session()
sess.run(init)

## we do it big dawg style and run each training step in our model 2000 times

for i in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
##Checking our work y, is our model's prediction. y_, is the correct answer.
##tf.argmax is an extremely useful function which gives you the index of the highest entry
##in a tensor along some axis.

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

##if our prediction is correct then correct_prediction will come up true(1), incorrect is 
##false(0). This next line of code gets the float average of the predictions of all our 
##test runs.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##then we print everything out and pray
 
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
