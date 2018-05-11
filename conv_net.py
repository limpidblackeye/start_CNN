from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import cv2
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import numpy as np
import tensorflow as tf
import sys
import argparse

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin


# configs
FLAGS = tf.app.flags.FLAGS
# mode
tf.app.flags.DEFINE_boolean('is_training', True, 'training or testing')
# data
tf.app.flags.DEFINE_string('root_dir', '../data', 'data root dir')
tf.app.flags.DEFINE_string('dataset', 'dset1', 'dset1 or dset2')
tf.app.flags.DEFINE_integer('n_label', 65, 'number of classes')
# trainig
tf.app.flags.DEFINE_integer('batch_size', 64, 'mini batch for a training iter')
tf.app.flags.DEFINE_string('save_dir', './checkpoints', 'dir to the trained model')
# test
tf.app.flags.DEFINE_string('my_best_model', './checkpoints/model.ckpt-1000', 'for test')

'''TODO: you may add more configs such as base learning rate, max_iteration,
display_iteration, valid_iteration and etc. '''


################ some parameters #################
IMAGE_SIZE = 128
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 65
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = None # 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
FLAGS = None
# tf.float32

#################################

################ define dataset #################

class DataSet(object):
    '''
    Args:
        data_aug: False for valid/testing.
        shuffle: true for training, False for valid/test.
    '''
    def __init__(self, root_dir, dataset, sub_set, batch_size, n_label,
                 data_aug=False, shuffle=True):
        np.random.seed(0)
        self.data_dir = os.path.join(root_dir, dataset, sub_set)
        self.batch_size = batch_size
        self.n_label = n_label
        self.data_aug = data_aug
        self.shuffle = shuffle
        self.xs, self.ys = self.load_data(root_dir, dataset, sub_set, n_label)
        self._num_examples = len(self.xs)
        self.init_epoch()

    def load_data(self):
        '''Fetch all data into a list'''
        '''TODO: 1. You may make it more memory efficient if there is a OOM problem on
        you machine. 2. You may use data augmentation tricks.'''
        xs = []
        ys = []
        label_dirs = os.listdir(self.data_dir)
        label_dirs.sort()
        for _label_dir in label_dirs:
            print('loaded {}'.format(_label_dir))
            category = int(_label_dir[5:])
            label = np.zeros(self.n_label)
            label[category] = 1
            imgs_name = os.listdir(os.path.join(self.data_dir, _label_dir))
            imgs_name.sort()
            for img_name in imgs_name:
                # im_ar = cv2.imread(os.path.join(self.data_dir, _label_dir, img_name))
                # im_ar = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)
                im_ar = imread(os.path.join(self.data_dir, _label_dir, img_name))
                im_ar = np.asarray(im_ar)
                im_ar = self.preprocess(im_ar)
                xs.append(im_ar)
                ys.append(label)
        return xs, ys

    def preprocess(self, im_ar):
        '''Resize raw image to a fixed size, and scale the pixel intensities.'''
        '''TODO: you may add data augmentation methods.'''
        # im_ar = cv2.resize(im_ar, (224, 224))
        image = resize(im_ar, (224, 224), mode='constant', preserve_range = True)
        im_ar = im_ar / 255.0
        return im_ar

    def next_batch(self):
        '''Fetch the next batch of images and labels.'''
        if not self.has_next_batch():
            return None
        print(self.cur_index)
        x_batch = []
        y_batch = []
        for i in xrange(self.batch_size):
            x_batch.append(self.xs[self.indices[self.cur_index+i]])
            y_batch.append(self.ys[self.indices[self.cur_index+i]])
        self.cur_index += self.batch_size
        return np.asarray(x_batch), np.asarray(y_batch)

    def has_next_batch(self):
        '''Call this function before fetching the next batch.
        If no batch left, a training epoch is over.'''
        start = self.cur_index
        end = self.batch_size + start
        if end > self._num_examples: return False
        else: return True

    def init_epoch(self):
        '''Make sure you would shuffle the training set before the next epoch.
        e.g. if not train_set.has_next_batch(): train_set.init_epoch()'''
        self.cur_index = 0
        self.indices = np.arange(self._num_examples)
        if self.shuffle:
            np.random.shuffle(self.indices)

#################################

################ define cnn model #################

class Model(object):
    def __init__(self):
        '''TODO: construct your model here.'''
        # Placeholders for input ims and labels
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        self.train_data_node = tf.placeholder(data_type(),shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        self.train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
        self.eval_data = tf.placeholder(data_type(),shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

        # Construct model
        self.logits = construct_model()
        self.prediction = tf.nn.softmax(self.logits)

        # Define loss and optimizer
        self.loss = tf.constant(0.0)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # init a tf session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        self.sess.run(init)

    def construct_model(self):
        '''TODO: Your code here.'''

        ################ declare variables #################
        # This is where training samples and labels are fed to the graph.

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when we call:
        # {tf.global_variables_initializer().run()}
        # 5x5 filter, depth 32.
        conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1,seed=SEED, dtype=data_type()))
        conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
        conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1,seed=SEED, dtype=data_type()))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
        # fully connected, depth 512.
        fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],stddev=0.1,seed=SEED,dtype=data_type()))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
        fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],stddev=0.1,seed=SEED,dtype=data_type()))
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))
        #################################

        ################ define network #################
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(self.train_data_node,conv1_weights,strides=[1, 1, 1, 1],padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        conv = tf.nn.conv2d(pool,conv2_weights,strides=[1, 1, 1, 1],padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        logits = tf.matmul(hidden, fc2_weights) + fc2_biases
        return logits

    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(eval_prediction,feed_dict={self.eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(eval_prediction,feed_dict={self.eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def train(self, ims, labels):
        '''TODO: Your code here.'''
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.train_labels_node, logits=logits))
        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0, dtype=data_type())
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
          0.01,                # Base learning rate.
          batch * BATCH_SIZE,  # Current index into the dataset.
          train_size,          # Decay step.
          0.95,                # Decay rate.
          staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)

        # Predictions for the current training minibatch.
        train_prediction = tf.nn.softmax(logits)
        # Small utility function to evaluate a dataset by feeding batches of data to
        # {eval_data} and pulling the results from {eval_predictions}.
        # Saves memory and enables this to run on smaller GPUs.

        # Create a local session to run the training.
        start_time = time.time()
        with tf.Session() as sess:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()
            print('Initialized!')
            # Loop through training steps.
            for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
                # This dictionary maps the batch data (as a np array) to the
                # node in the graph it should be fed to.
                feed_dict = {self.train_data_node: batch_data,self.train_labels_node: batch_labels}
                # Run the optimizer to update weights.
                sess.run(optimizer, feed_dict=feed_dict)
                # print some extra information once reach the evaluation frequency
                if step % EVAL_FREQUENCY == 0:
                    # fetch some extra nodes' data
                    l, lr, predictions = sess.run([loss, learning_rate, train_prediction],feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                    print('Validation error: %.1f%%' % error_rate(self.eval_in_batches(validation_data, sess), validation_labels))
                    sys.stdout.flush()
                    self.save(self.checkpoint_path, step)

            # valid_error = error_rate(self.eval_in_batches(valid_data, sess), valid_labels)
            # print('Validation error: %.1f%%' % valid_error)
     
        self.loss = l
        return self.loss

    def valid(self, ims, labels):
        '''TODO: Your code here.'''
        # loss, acc = sess.run(Model[:-1], feed_dict={x: xTe, y: yTe})
        # print final result
        test_error = error_rate(self.eval_in_batches(ims, sess), labels)
        print('Test error: %.1f%%' % test_error)
        # Predictions for the test and validation, which we'll compute less often.
        predictions = tf.nn.softmax(model(self.eval_data))
        return self.predictions

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)

    def load(self):
        print('load model:', FLAGS.my_best_model)
        self.saver.restore(self.sess, FLAGS.my_best_model)

#################################

################ define train and test #################

def train_wrapper(model):
    '''Data loader'''
    train_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'train',
                        FLAGS.batch_size, FLAGS.n_label,
                        data_aug=False, shuffle=True)
    valid_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'valid',
                        FLAGS.batch_size, FLAGS.n_label,
                        data_aug=False, shuffle=False)
    train_data, train_labels = train_set.load_data()
    valid_data, valid_labels = valid_set.load_data()
    '''create a tf session for training and validation
    TODO: to run your model, you may call model.train(), model.save(), model.valid()'''
    # declare model
    with tf.Session() as sess:
        model=Model(sess)
        model.train(train_data, train_labels)
        model.valid(valid_data, valid_labels)
   
def test_wrapper(model):
    '''Finish this function so that TA could test your code easily.'''    
    test_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'test',
                       FLAGS.batch_size, FLAGS.n_label,
                       data_aug=False, shuffle=False)
    test_data, test_labels = test_set.load_data()
    '''TODO: Your code here.'''
        # load checkpoints
    with tf.Session() as sess:
        model=Model(sess)
        if model.load(model.checkpoint_path, model.dataset_name):
            print("[*] SUCCESS to load model for %s." % model.dataset_name)
        else:
            print("[!] Failed to load model for %s." % model.dataset_name)
            sys.exit(1)
        model.valid(test_data, test_labels)

def main(argv=None):
    print('Initializing models')
    model = Model()
    if FLAGS.is_training:
        train_wrapper(model)
    else:
        test_wrapper(model)

#################################
if __name__ == '__main__':
    tf.app.run()

