import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import math

# configs
FLAGS = tf.app.flags.FLAGS
# mode
tf.app.flags.DEFINE_boolean('is_training', True, 'training or testing')
# data
tf.app.flags.DEFINE_string('root_dir', './datasets/homework', 'data root dir')
tf.app.flags.DEFINE_string('dataset', 'dset1', 'dset1 or dset2')
tf.app.flags.DEFINE_integer('n_label', 65, 'number of classes')
# trainig
tf.app.flags.DEFINE_integer('batch_size', 64, 'mini batch for a training iter')
tf.app.flags.DEFINE_string('save_dir', './checkpoints', 'dir to the trained model')
# test
tf.app.flags.DEFINE_string('my_best_model', './checkpoints/model.ckpt-1000', 'for test')

'''TODO: you may add more configs such as base learning rate, max_iteration,
display_iteration, valid_iteration and etc. '''


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
            print 'loaded {}'.format(_label_dir)
            category = int(_label_dir[5:])
            label = np.zeros(self.n_label)
            label[category] = 1
            imgs_name = os.listdir(os.path.join(self.data_dir, _label_dir))
            imgs_name.sort()
            for img_name in imgs_name:
                im_ar = cv2.imread(os.path.join(self.data_dir, _label_dir, img_name))
                im_ar = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)
                im_ar = np.asarray(im_ar)
                im_ar = self.preprocess(im_ar)
                xs.append(im_ar)
                ys.append(label)
        return xs, ys

    def preprocess(self, im_ar):
        '''Resize raw image to a fixed size, and scale the pixel intensities.'''
        '''TODO: you may add data augmentation methods.'''
        im_ar = cv2.resize(im_ar, (224, 224))
        im_ar = im_ar / 255.0
        return im_ar

    def next_batch(self):
        '''Fetch the next batch of images and labels.'''
        if not self.has_next_batch():
            return None
        print self.cur_index
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
        loss = tf.constant(0.0)

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
                yOut = None
        """
            I tried several times
            I find the response normalization does work well and it takes a lot of time which influence the train speed
            Finally discard it
        """
        #Define Convolutional layer
        filter1 = tf.get_variable(name = "wConv", shape = [3, 3, 3, 64])
        bConv1 = tf.get_variable(name = "bConv", shape = [64])

        conv1 = tf.nn.conv2d(x, filter1, strides = [1, 1, 1, 1], padding = 'SAME') + bConv1
        Rconv1 = tf.nn.relu(conv1) # output size = (32-3 + 2)/1 +1 =32 )

        #Define convolutional layer
        filter2 = tf.get_variable(name = "wCvon2", shape = [3,3,64,64])
        bConv2 = tf.get_variable(name = "bConv2", shape=[64])

        conv2 = tf.nn.conv2d(Rconv1, filter2, strides = [1, 1, 1, 1], padding = 'SAME') + bConv2
        Rconv2 = tf.nn.relu(conv2)#output size = (32-3+2) /1 +1 =32

        #Define maxpool layer
        h = tf.nn.max_pool(Rconv2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        #output size = (32-2) /2 +1 = 16

        #Define Convolutional layer
        filter3 = tf.get_variable(name = "wConv3", shape = [3, 3, 64, 64])
        bConv3 = tf.get_variable(name = "bConv3", shape = [64])

        conv3 = tf.nn.conv2d(h, filter3, strides = [1, 1, 1, 1], padding = 'SAME') + bConv3
        Rconv3 = tf.nn.relu(conv3) # output size = (16-3 + 2)/1 +1 =16

        #Define convolutional layer
        filter4 = tf.get_variable(name = "wCvon4", shape = [3,3,64,64])
        bConv4 = tf.get_variable(name = "bConv4", shape=[64])

        conv4 = tf.nn.conv2d(Rconv3, filter4, strides = [1, 1, 1, 1], padding = 'SAME') + bConv4
        Rconv4 = tf.nn.relu(conv4)#output size = (16-3+2) /1 +1 =16

        # Define maxpool layer
        h = tf.nn.max_pool(Rconv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # output size = (16-2) /2 +1 = 8

        #Define hidden layer
        w1 = tf.get_variable(name = "w1", shape = [4096, 512],) # parameter to learn : 8*8*64 = 4096
        b1 = tf.get_variable(name = "b1", shape = [512])
        h_shift = tf.reshape(h, [-1, 4096])
        hidden_output = tf.matmul(h_shift, w1) + b1
        hidden_output = tf.nn.relu(hidden_output)
        hidden_output1 = tf.nn.dropout(hidden_output, 0.8)#dropout to avoid overfitting

        w2 = tf.get_variable(name = "w2", shape=[512,64])
        b2 = tf.get_variable(name = "b2", shape = [64])
        fc = tf.matmul(hidden_output1, w2) + b2
        fc = tf.nn.relu(fc)

        #Define output layer
        w3 = tf.get_variable(name = "w3", shape = [64, 10])
        b3 = tf.get_variable(name = 'b3', shape = [10])
        yOut = tf.matmul(fc, w3) + b3

        # Define Loss
        totalLoss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), logits=yOut) + 5e-4 * tf.nn.l2_loss(w1) + 5e-4 * tf.nn.l2_loss(w2) + 5e-4 * tf.nn.l2_loss(w3)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate= 0.001)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]
        return logits

    def train(self, ims, labels):
        '''TODO: Your code here.'''
        return self.loss

    def valid(self, ims, labels):
        '''TODO: Your code here.'''
        return self.predictions

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)

    def load(self):
        print('load model:', FLAGS.my_best_model)
        self.saver.restore(self.sess, FLAGS.my_best_model)


def train_wrapper(model):
    '''Data loader'''
    train_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'train',
                        FLAGS.batch_size, FLAGS.n_label,
                        data_aug=False, shuffle=True)
    valid_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'valid',
                        FLAGS.batch_size, FLAGS.n_label,
                        data_aug=False, shuffle=False)
    '''create a tf session for training and validation
    TODO: to run your model, you may call model.train(), model.save(), model.valid()'''


def test_wrapper(model):
    '''Finish this function so that TA could test your code easily.'''    
    test_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'test',
                       FLAGS.batch_size, FLAGS.n_label,
                       data_aug=False, shuffle=False)
    '''TODO: Your code here.'''


def main(argv=None):
    print('Initializing models')
    model = Model()
    if FLAGS.is_training:
        train_wrapper(model)
    else:
        test_wrapper(model)


if __name__ == '__main__':
    tf.app.run()

