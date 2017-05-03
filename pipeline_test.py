import os
import sys
import time
import shutil
import tensorflow as tf

def _prepare_graph():
    tf.reset_default_graph()
    tf.set_random_seed(23)


class Model(object):
    
    def __init__(self, inputs, targets):
        self._gs = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)

        self._inputs = inputs
        self._targets = targets
        
        with tf.variable_scope('Layer1') as scope:
            w1 = tf.get_variable('w', shape=[1024, 1024])
            b1 = tf.get_variable('b', shape=[1024])
            z1 = tf.matmul(self._inputs, w1) + b1
            y1 = tf.nn.relu(z1, name='activation')

        with tf.variable_scope('Layer2') as scope:
            w2 = tf.get_variable('w', shape=[1024, 1])
            b2 = tf.get_variable('b', shape=[1])
            logits = tf.matmul(y1, w2) + b2
            self._predicts = tf.cast(logits > 0, tf.int32)

        with tf.variable_scope('Loss'):
            labels = tf.cast(self._targets, tf.float32)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits)
            self._loss_op = tf.reduce_mean(losses, name='loss')

        with tf.variable_scope('BackProp'):
            optimizer = tf.train.GradientDescentOptimizer(0.1)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads_and_vars = optimizer.compute_gradients(
                loss=self._loss_op, var_list=trainable_vars)
            self._train_op = optimizer.apply_gradients(
                grads_and_vars=grads_and_vars, 
                global_step=self._gs,
                name="train_op")
            
        with tf.variable_scope('Accuracy'):
            self._accuracy_op = tf.reduce_mean(
                tf.cast(tf.equal(self._predicts, self._targets), tf.float32),
                name='accuracy')
            
            
    @property
    def global_step(self):
        return self._gs

    @property
    def inputs(self):
        return self._inputs
    
    @property
    def targets(self):
        return self._targets
    
    @property
    def predictions(self):
        return self._predicts
    
    @property
    def loss_op(self):
        return self._loss_op
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def accuracy_op(self):
        return self._accuracy_op


def get_batch_tensors(batch_size=128):
    data = tf.random_normal([batch_size, 1024], mean=0, stddev=1)
    labels = tf.cast(tf.reduce_sum(data, axis=1, keep_dims=True) > 0, tf.int32)
    return data, labels


def slow():
    _prepare_graph()
    with tf.variable_scope('Input') as scope:
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 1024])
        targets = tf.placeholder(dtype=tf.int32, shape=[None, 1])
    model = Model(inputs, targets)
    STEPS = 100
    CHECK_EVERY = 10

    startTime = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(STEPS):
            actual_inputs, actual_targets = sess.run(list(get_batch_tensors(128)))
            fetches = [model.train_op]
            if i % CHECK_EVERY == 0:
                fetches = fetches + [model.loss_op, model.accuracy_op]
            results = sess.run(
                fetches=fetches,
                feed_dict={
                    model.inputs: actual_inputs,
                    model.targets: actual_targets
                })
            if len(results) > 1:
                print('iter:%d - loss:%f - accuracy:%f' % (i, results[1], results[2]))
    print("Time taken: %f" % (time.time() - startTime))


def fast():
    _prepare_graph()

    NUM_THREADS = 2
    FIFOQUEUE_NAME = 'FIFOQueue'

    with tf.variable_scope('Input') as scope:
        inputs, targets = get_batch_tensors(128)
        queue = tf.FIFOQueue(capacity=5, dtypes=[tf.float32, tf.int32], name=FIFOQUEUE_NAME)
        enqueue_op = queue.enqueue((inputs, targets))
        queue_runner = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
        tf.train.add_queue_runner(queue_runner)
        inputs, targets = queue.dequeue()
    model = Model(inputs, targets)

    STEPS = 100
    CHECK_EVERY = 10

    startTime = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(STEPS):        
            fetches = [model.train_op]
            if i % CHECK_EVERY == 0:
                fetches = fetches + [model.loss_op, model.accuracy_op]
            results = sess.run(fetches)
            if len(results) > 1:
                print('iter:%d - loss:%f - accuracy:%f' % (i, results[1], results[2]))

        coord.request_stop()
        coord.join(threads)
    print("Time taken: %f" % (time.time() - startTime))


if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == 'slow':
        slow()
    elif arg == 'fast':
        fast()
    else:
        pass