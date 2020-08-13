# Distributed Tensorflow 1.6 Example (DEPRECATED) python 3.6

Using data parallelism with shared model parameters while updating parameters asynchronous. See comment for some changes to make the parameter updates synchronous (not sure if the synchronous part is implemented correctly though).

Trains a simple sigmoid Neural Network on MNIST for 20 epochs on three machines using one parameter server. The goal was not to achieve high accuracy but to get to know tensorflow.

# mnist_tfrecords.py 


