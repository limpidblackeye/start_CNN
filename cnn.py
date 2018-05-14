import os
# import cv2
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import numpy as np
import tensorflow as tf

from six.moves import urllib
from six.moves import xrange 

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
tf.app.flags.DEFINE_string('my_best_model', './checkpoints/my_best_model.model.ckpt', 'for test')
tf.app.flags.DEFINE_float("learning_rate", 0.01, "The learning rate")

'''TODO: you may add more configs such as base learning rate, max_iteration,
display_iteration, valid_iteration and etc. '''

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
                # im_ar = cv2.imread(os.path.join(self.data_dir, _label_dir, img_name))
                # im_ar = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)
                im_ar = imread(os.path.join(self.data_dir, _label_dir, img_name))
                # print(os.path.join(self.data_dir, _label_dir, img_name))
                im_ar = np.asarray(im_ar)
                im_ar = self.preprocess(im_ar)
                xs.append(im_ar)
                ys.append(list(label))
        return xs, ys

    def preprocess(self, im_ar):
        '''Resize raw image to a fixed size, and scale the pixel intensities.'''
        '''TODO: you may add data augmentation methods.'''
        # im_ar = cv2.resize(im_ar, (224, 224))
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
        # print("np.asarray(x_batch).shape:",np.asarray(x_batch).shape)
        # print("np.asarray(y_batch).shape:",np.asarray(y_batch).shape)
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
        self.X = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
        self.Y = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.n_label])
        self.YShape = tf.shape(self.Y)
        self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, FLAGS.n_label]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([FLAGS.n_label]))
        }

        # Construct model
        self.logits = self.construct_model()
        print("logits' shape is:", tf.shape(self.logits))
        print("Y's shape is", tf.shape(self.Y))
        #self.logits=tf.Print(self.logits,["logits", self.logits.shape()])
        self.shapelogits = tf.shape(self.logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Evaluate model
        self.prediction = tf.argmax(tf.nn.softmax(self.logits),1)
        self.correct_pred = tf.equal(self.prediction, tf.argmax(self.Y, 1))
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
    
    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

    def construct_model(self):
        '''TODO: Your code here.'''
        # Convolution Layer
        conv1 = self.conv2d(self.X, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)
        self.shapeconv1 = tf.shape(conv1)
        self.XShape = tf.shape(self.X)
    
        # Convolution Layer
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)
        self.shapeconv2 = tf.shape(conv2)
    
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        # fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.contrib.layers.flatten(conv2)
        # fc1 = tf.reshape(conv2, [FLAGS.batch_size, 7*7*64])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.keep_prob)
        self.shapefc1 = tf.shape(fc1)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

    def train(self, ims, labels):
        '''TODO: Your code here.'''
        with tf.device('/gpu:1'):
        	logits, label, loss, _, acc  = self.sess.run([self.logits, self.Y, self.loss_op, self.train_op, self.accuracy], feed_dict={self.X: ims, self.Y: labels, self.keep_prob: 0.8})
        #print "The shape is:"
        #print "logits=", logits
        #print "label=", label
        	return loss, acc
        # with tf.Session() as sess:
            #sess.run(self.init)
            #logits, label, loss, _  = self.sess.run([self.logits, self.Y, self.loss_op, self.train_op], feed_dict={self.X: ims, self.Y: labels, self.keep_prob: 0.8})
           # print self.logits.eval()
        #print self.sess.run([self.XShape, self.YShape, self.shapeconv1, self.shapeconv2, self.shapefc1, self.shapelogits], feed_dict={self.X: ims, self.Y: labels, self.keep_prob: 0.8})

    def valid(self, ims, labels):
        '''TODO: Your code here.'''
        prediction, loss, acc = self.sess.run([self.prediction, self.loss_op, self.accuracy], feed_dict={self.X: ims,
                                                                 self.Y: labels,
                                                                 self.keep_prob: 1.0})
        return prediction, loss, acc

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
    best_accuracy = 0
    for i in xrange(10000):
        if not train_set.has_next_batch():
            train_set.init_epoch()
        train_img, train_label = train_set.next_batch()
        if len(train_img == FLAGS.batch_size):
            loss_train, acc_train = model.train(train_img, train_label)
        if i % 1000 == 0:
            pre = list()
            tot_acc = 0
            tot_input = 0
            while valid_set.has_next_batch():
                valid_img, valid_label = valid_set.next_batch()
                predictions, loss, acc = model.valid(valid_img, valid_label)
                tot_acc += acc * len(valid_img)
                tot_input += len(valid_img)
            print("current loss is:", loss_train)
            print(", acc is:", acc_train)
            print(", iter=", i)
            print("tot_acc=", tot_acc)
            print("tot_input=", tot_input)
            acc = tot_acc / tot_input
            valid_set.init_epoch()
            print("current accuracy is:", acc)
            if acc > best_accuracy:
                model.save(i)
                best_accuracy = acc


def test_wrapper(model):
    '''for the TA to test your model'''
    pass


def main(argv=None):
    print('Initializing models')
    model = Model()
    if FLAGS.is_training:
        train_wrapper(model)
    else:
        test_wrapper(model)


if __name__ == '__main__':
    tf.app.run()


