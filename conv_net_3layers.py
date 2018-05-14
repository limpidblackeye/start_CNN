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
import time

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

# configs
FLAGS = tf.app.flags.FLAGS
# mode
tf.app.flags.DEFINE_boolean('is_training', True, 'training or testing')
# data
# tf.app.flags.DEFINE_string('root_dir', '../data_fortest_10label', 'data root dir')
tf.app.flags.DEFINE_string('root_dir', '../data', 'data root dir')
tf.app.flags.DEFINE_string('dataset', 'dset1', 'dset1 or dset2')
tf.app.flags.DEFINE_integer('n_label', 65, 'number of classes')
# trainig
tf.app.flags.DEFINE_integer('batch_size', 16, 'mini batch for a training iter')
tf.app.flags.DEFINE_string('save_dir', '../checkpoints_conv_net_3layers', 'dir to the trained model')
# test
tf.app.flags.DEFINE_string('my_best_model', '../checkpoints_conv_net_3layers/model.ckpt-8100', 'for test')

'''TODO: you may add more configs such as base learning rate, max_iteration,
display_iteration, valid_iteration and etc. '''

################ some parameters #################
IMAGE_SIZE = 224
NUM_CHANNELS = 3
# PIXEL_DEPTH = 255
NUM_LABELS = FLAGS.n_label
# VALIDATION_SIZE = 5000  # Size of the validation set.
# SEED = None # 66478  # Set to None for random seed.
BATCH_SIZE = FLAGS.batch_size
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = BATCH_SIZE
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
# FLAGS = None
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
        self.xs, self.ys = self.load_data()
        self._num_examples = len(self.xs)
        self.init_epoch()

    def load_data(self):
        '''Fetch all data into a list'''
        '''TODO: 1. You may make it more memory efficient if there is a OOM problem on
        you machine. 2. You may use data augmentation tricks.'''
        xs = []
        ys = []
        label_dirs = os.listdir(self.data_dir)
        a_label_dirs = [int(label_dirs[k][5:]) for k in range(len(label_dirs))]
        label_dirs = [x for (y,x) in sorted(zip(a_label_dirs,label_dirs))]
        # label_dirs.sort()
        label_index = 0
        for _label_dir in label_dirs:
            # print(label_index)
            print('loaded {}'.format(_label_dir))
            label = np.zeros(self.n_label)
            label[label_index] = 1
            label_index += 1
            imgs_name = os.listdir(os.path.join(self.data_dir, _label_dir))
            imgs_name.sort()
            for img_name in imgs_name:
                im_ar = imread(os.path.join(self.data_dir, _label_dir, img_name))
                im_ar = np.asarray(im_ar)
                im_ar = self.preprocess(im_ar)
                xs.append(im_ar)
                ys.append(list(label))
        return xs, ys

    def preprocess(self, im_ar):
        '''Resize raw image to a fixed size, and scale the pixel intensities.'''
        im_ar = resize(im_ar, (224, 224), mode='constant', preserve_range = True)
        im_ar = im_ar / 255.0
        return im_ar

    def next_batch(self):
        '''Fetch the next batch of images and labels.'''
        if not self.has_next_batch():
            return None
        # print(self.cur_index)
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
        self.train_data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        self.train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE, NUM_LABELS))
        self.eval_data = tf.placeholder(tf.float32,shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        self.drop_out_rate = tf.placeholder(tf.float32)

        # define weight and bias
        self.conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 16], stddev=0.001, dtype=tf.float32))
        self.conv1_biases = tf.Variable(tf.random_normal([16], dtype=tf.float32))
        self.conv1_beta = tf.Variable(tf.zeros([16]))
        self.conv1_gamma = tf.Variable(tf.truncated_normal([16],stddev=0.01))

        self.conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1, dtype=tf.float32))
        self.conv2_biases = tf.Variable(tf.random_normal([32], dtype=tf.float32))
        self.conv2_beta = tf.Variable(tf.zeros([32]))
        self.conv2_gamma = tf.Variable(tf.truncated_normal([32],stddev=0.01))

        self.conv3_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, dtype=tf.float32))
        self.conv3_biases = tf.Variable(tf.random_normal([64], dtype=tf.float32))
        self.conv3_beta = tf.Variable(tf.zeros([64]))
        self.conv3_gamma = tf.Variable(tf.truncated_normal([64],stddev=0.1))

        # fully connected, depth 512.
        self.fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 16, 512],stddev=0.1,dtype=tf.float32))
        self.fc1_biases = tf.Variable(tf.random_normal([512], dtype=tf.float32))
        self.fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],stddev=0.1,dtype=tf.float32))
        self.fc2_biases = tf.Variable(tf.random_normal([NUM_LABELS], dtype=tf.float32))

        # Construct model
        self.logits = self.construct_model()
    
        # Define loss and optimizer
        # self.loss = tf.constant(0.0)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels_node, logits=self.logits))
        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) + tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        self.loss += 5e-4 * regularizers

        # optimizer
        # controls the learning rate decay.
        self.batch = tf.Variable(0, dtype=tf.float32)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        # self.learning_rate = tf.train.exponential_decay(
        #   0.1,                # Base learning rate.
        #   self.batch * BATCH_SIZE,  # Current index into the dataset.
        #   self.train_labels_node.shape[0],         # Decay step.
        #   0.999,                # Decay rate.
        #   staircase=True)
        self.learning_rate = tf.train.piecewise_constant(
            self.batch,
            boundaries = [2000.0,2001.0,3000.0,4000.0],
            values = [0.001,0.005,0.0001,0.0001,0.00001]
            )
        # Use simple momentum for the optimization.
        # self.learning_rate = tf.Variable(0.01, dtype=tf.float32)
        # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9).minimize(self.loss, global_step=self.batch)
        if tf.greater(tf.Variable(6000.0,dtype=tf.float32),self.batch)==tf.Variable(True,dtype=tf.bool):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.batch)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.batch)

        # Evaluate model
        self.prediction = tf.nn.softmax(self.logits)
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.train_labels_node, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # init a tf session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        self.init = tf.global_variables_initializer()
        self.configProt = tf.ConfigProto()
        self.configProt.gpu_options.allow_growth = True
        self.configProt.allow_soft_placement = True
        self.sess = tf.Session(config=self.configProt)
        self.sess.run(self.init)

    def construct_model(self):
        ################ define network #################
        """The Model definition."""

        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        
        # conv1
        conv1 = tf.nn.conv2d(self.train_data_node,self.conv1_weights,strides=[1, 1, 1, 1],padding='SAME')
        mean1, var1 = tf.nn.moments(conv1, axes=[0,1,2])
        batch_norm1 = tf.nn.batch_norm_with_global_normalization(
            conv1, mean1, var1, self.conv1_beta, self.conv1_gamma, 0.001,
            scale_after_normalization=True)
        relu1 = tf.nn.relu(tf.nn.bias_add(batch_norm1, self.conv1_biases))
        pool1 = tf.nn.max_pool(relu1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    
        # conv2
        conv2 = tf.nn.conv2d(pool1,self.conv2_weights,strides=[1, 1, 1, 1],padding='SAME')
        mean2, var2 = tf.nn.moments(conv2, axes=[0,1,2])
        batch_norm2 = tf.nn.batch_norm_with_global_normalization(
            conv2, mean2, var2, self.conv2_beta, self.conv2_gamma, 0.001,
            scale_after_normalization=True)
        relu2 = tf.nn.relu(tf.nn.bias_add(batch_norm2, self.conv2_biases))
        # pool2 = tf.nn.max_pool(relu2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        pool2 = tf.nn.avg_pool(relu2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

        # conv3
        conv3 = tf.nn.conv2d(pool2,self.conv3_weights,strides=[1, 1, 1, 1],padding='SAME')
        mean3, var3 = tf.nn.moments(conv3, axes=[0,1,2])
        batch_norm3 = tf.nn.batch_norm_with_global_normalization(
            conv3, mean3, var3 , self.conv3_beta, self.conv3_gamma, 0.001,
            scale_after_normalization=True)
        relu3 = tf.nn.relu(tf.nn.bias_add(batch_norm3, self.conv3_biases))
        # pool3 = tf.nn.max_pool(relu3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        pool3 = tf.nn.avg_pool(relu3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        # pool_shape = pool.get_shape().as_list()
        # reshape = tf.reshape(pool,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        reshape = tf.contrib.layers.flatten(pool3)
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        hidden = tf.nn.dropout(hidden, self.drop_out_rate)
        logits = tf.add(tf.matmul(hidden, self.fc2_weights), self.fc2_biases)
        return logits

    def train(self, ims, labels):
        with tf.device('/gpu:0'):
        # with tf.Session() as sess:
        #     sess.run(self.init)
            logit, label, loss, opti, acc, lr  = self.sess.run([self.logits,self.train_labels_node, self.loss, self.optimizer,  self.accuracy, self.learning_rate], feed_dict={self.train_data_node: ims, self.train_labels_node: labels, self.drop_out_rate: 0.8})
            return loss, acc, lr

    def valid(self, ims, labels):
        prediction, loss, acc = self.sess.run([self.prediction, self.loss, self.accuracy], feed_dict={self.train_data_node: ims, self.train_labels_node: labels, self.drop_out_rate: 1})
        return prediction, loss, acc

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

    # Create a local session to run the training.
    num_epochs = NUM_EPOCHS
    train_size = train_set._num_examples
    print("train_size:",train_size)
    start_time = time.time()
    # Run all the initializers to prepare the trainable parameters.
    # tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    best_accuracy = 0
    # for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
    for step in xrange(10000):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        if not train_set.has_next_batch():
            train_set.init_epoch()
        train_data, train_labels = train_set.next_batch()
        loss, acc, lr = model.train(train_data, train_labels)

        # print some extra information once reach the evaluation frequency
        if step % EVAL_FREQUENCY == 0:
            tot_acc = 0
            tot_input = 0
            while valid_set.has_next_batch():
                valid_data, valid_labels = valid_set.next_batch()
                _, loss_val, acc_val = model.valid(valid_data, valid_labels)
                tot_acc += acc_val * len(valid_data)
                tot_input += len(valid_data)
            acc_val = tot_acc / tot_input
            print("tot_acc:",tot_acc)
            print("tot_input",tot_input)
            valid_set.init_epoch()

            # l, lr, predictions = sess.run([loss, learning_rate, train_prediction],feed_dict=feed_dict)
            elapsed_time = time.time() - start_time
            start_time = time.time()

            print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (loss, lr))
            
            print('Minibatch accuracy: %.3f' % acc)
            print('Validation accuracy: %.3f' % acc_val)
            if acc_val > best_accuracy:
                best_accuracy = acc_val
                model.save(step)
    print('Final validation best_accuracy:%.3f' % best_accuracy)

def test_wrapper(model):
    '''Finish this function so that TA could test your code easily.'''    
    test_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'test',
                       FLAGS.batch_size, FLAGS.n_label,
                       data_aug=False, shuffle=False)
    '''TODO: Your code here.'''
        # load checkpoints
    if model.load():
        print("[*] SUCCESS to load model")
    else:
        print("[!] Failed to load model!")
        sys.exit(1)
    tot_acc = 0
    tot_input = 0
    while test_set.has_next_batch():
        test_data, test_labels = test_set.next_batch()
        _, loss_val, acc_val = model.valid(test_data, test_labels)
        tot_acc += acc_val * len(test_data)
        tot_input += len(test_data)
    acc_val = tot_acc / tot_input
    print("tot_acc:",tot_acc)
    print("tot_input",tot_input)

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

