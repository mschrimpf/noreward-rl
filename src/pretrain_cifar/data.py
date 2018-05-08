import glob
import os
import pickle
import shutil
import sys
from abc import abstractmethod, ABCMeta
from random import randint

import numpy as np
import tensorflow as tf


class Dataset:
    __metaclass__ = ABCMeta

    num_threads = 8
    output_buffer_size = 1024

    list_labels = range(0)
    num_images_training = 0
    num_images_test = 0

    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def get_data_trainval(self):
        # Returns images training & labels
        pass

    @abstractmethod
    def get_data_test(self):
        # Returns images training & labels
        pass

    @abstractmethod
    def preprocess_image(self, image):
        # Returns images training & labels
        pass

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Write one TF records file
    def write_tfrecords(self, tfrecords_path, set_name, addrs, labels):

        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(tfrecords_path + set_name + '.tfrecords')

        for i in range(len(addrs)):
            # print how many images are saved every 1000 images
            if not i % 1000:
                print('Data: {}/{}'.format(i, len(addrs)))
                sys.stdout.flush()

            # Create a feature
            feature = {set_name + '/label': self._int64_feature(labels[i]),
                       set_name + '/image': self._bytes_feature(addrs[i].tostring()),
                       set_name + '/width': self._int64_feature(32),
                       set_name + '/height': self._int64_feature(32)
                       }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    # Create all TFrecords files
    def create_tfrecords(self):

        if not self.opt.dataset.reuse_TFrecords:
            tfrecords_path = self.opt.log_dir_base + self.opt.name + '/data/'
        else:
            tfrecords_path = self.opt.log_dir_base + self.opt.dataset.reuse_TFrecords_path + '/data/'
            print("REUSING TFRECORDS")

        if os.path.isfile(tfrecords_path + 'test.tfrecords'):
            return 0

        if os.path.isdir(tfrecords_path):
            shutil.rmtree(tfrecords_path)

        os.makedirs(tfrecords_path)

        print("CREATING TFRECORDS")
        print(self.opt.dataset.dataset_path)

        train_addrs, train_labels, val_addrs, val_labels = self.get_data_trainval()
        app = self.opt.dataset.transfer_append_name
        self.write_tfrecords(tfrecords_path, 'train' + app, train_addrs, train_labels)
        self.write_tfrecords(tfrecords_path, 'val' + app, val_addrs, val_labels)

        test_addrs, test_labels = self.get_data_test()
        self.write_tfrecords(tfrecords_path, 'test' + app, test_addrs, test_labels)

    def delete_tfrecords(self):
        tfrecords_path = self.opt.log_dir_base + self.opt.name + '/data/'
        shutil.rmtree(tfrecords_path)

    def create_dataset(self, augmentation=False, standarization=False, set_name='train', repeat=False):
        app = self.opt.dataset.transfer_append_name
        set_name_app = set_name + app

        # Transforms a scalar string `example_proto` into a pair of a scalar string and
        # a scalar integer, representing an image and its label, respectively.
        def _parse_function(example_proto):
            features = {set_name_app + '/label': tf.FixedLenFeature((), tf.int64, default_value=1),
                        set_name_app + '/image': tf.FixedLenFeature((), tf.string, default_value=""),
                        set_name_app + '/height': tf.FixedLenFeature([], tf.int64),
                        set_name_app + '/width': tf.FixedLenFeature([], tf.int64)}
            parsed_features = tf.parse_single_example(example_proto, features)
            image = tf.decode_raw(parsed_features[set_name_app + '/image'], tf.uint8)
            image = tf.cast(image, tf.float32)
            S = tf.stack([tf.cast(parsed_features[set_name_app + '/height'], tf.int32),
                          tf.cast(parsed_features[set_name_app + '/width'], tf.int32), 3])
            image = tf.reshape(image, S)

            float_image = self.preprocess_image(augmentation, standarization, image)

            return float_image, parsed_features[set_name_app + '/label']

        # Creates a dataset that reads all of the examples from two files, and extracts
        # the image and label features.
        if not self.opt.dataset.reuse_TFrecords:
            tfrecords_path = self.opt.log_dir_base + self.opt.name + '/data/'
        else:
            tfrecords_path = self.opt.log_dir_base + self.opt.dataset.reuse_TFrecords_path + '/data/'

        filenames = [tfrecords_path + set_name_app + '.tfrecords']
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function, num_threads=self.num_threads, output_buffer_size=self.output_buffer_size)

        if repeat:
            dataset = dataset.repeat()  # Repeat the input indefinitely.

        return dataset.batch(self.opt.hyper.batch_size)


class Cifar10(Dataset):

    def __init__(self, opt):
        super(Cifar10, self).__init__(opt)

        self.num_threads = 8
        self.output_buffer_size = 1024

        self.list_labels = list(range(0, 10))
        self.num_images_training = 50000
        self.num_images_test = 10000

        self.num_images_epoch = self.opt.dataset.proportion_training_set * self.num_images_training
        self.num_images_val = self.num_images_training - self.num_images_epoch

        self.image_size = opt.ob_space[:-1]

        self.create_tfrecords()

    # Helper functions:
    def __unpickle(self, file_name):
        with open(file_name, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    # Virtual functions:
    def get_data_trainval(self):
        # read the 5 batch files of cifar
        addrs = []
        labels = []

        perm = np.random.permutation(32 * 32).astype("uint8")
        perm_x = (perm / 32).astype("uint8")
        perm_y = (perm % 32).astype("uint8")

        file_names = glob.glob(self.opt.dataset.dataset_path + "*_batch_*")
        for l in file_names:
            d = self.__unpickle(l)
            tmp = dict(d)
            X = tmp['data'].astype("uint8").reshape(10000, 3, 32, 32)
            if self.opt.dataset.scramble_data:
                X = X[:, :, perm_x, perm_y].reshape(10000, 3, 32, 32)
            X = X.transpose(0, 2, 3, 1)
            # X = tf.image.resize_images(X, self.image_size)
            [addrs.append(l) for l in X]
            if not self.opt.dataset.random_labels:
                [labels.append(l) for l in tmp['labels']]
            else:
                [labels.append(randint(0, 9)) for l in tmp['labels']]

            train_addrs = []
            train_labels = []
            val_addrs = []
            val_labels = []

            # Divide the data into 95% train, 5% validation
            [train_addrs.append(elem) for elem in addrs[0:int(self.opt.dataset.proportion_training_set * len(addrs))]]
            [train_labels.append(elem) for elem in labels[0:int(self.opt.dataset.proportion_training_set * len(addrs))]]

            [val_addrs.append(elem) for elem in addrs[int(self.opt.dataset.proportion_training_set * len(addrs)):]]
            [val_labels.append(elem) for elem in labels[int(self.opt.dataset.proportion_training_set * len(addrs)):]]

        return train_addrs, train_labels, val_addrs, val_labels

    def get_data_test(self):
        test_addrs = []
        test_labels = []
        file_names = glob.glob(self.opt.dataset.dataset_path + "test_batch")

        perm = np.random.permutation(32 * 32).astype("uint8")
        perm_x = (perm / 32).astype("uint8")
        perm_y = (perm % 32).astype("uint8")

        for l in file_names:
            d = self.__unpickle(l)
            tmp = dict(d)
            X = tmp['data'].astype("uint8").reshape(10000, 3, 32, 32)
            if self.opt.dataset.scramble_data:
                X = X[:, :, perm_x, perm_y].reshape(10000, 3, 32, 32)
            X = X.transpose(0, 2, 3, 1)

            [test_addrs.append(l) for l in X]
            if not self.opt.dataset.random_labels:
                [test_labels.append(l) for l in tmp['labels']]
            else:
                [test_labels.append(randint(0, 9)) for l in tmp['labels']]

        return test_addrs, test_labels

    def preprocess_image(self, augmentation, standarization, image):
        if augmentation:
            # Randomly crop a [height, width] section of the image.
            distorted_image = tf.random_crop(image, [self.opt.hyper.crop_size, self.opt.hyper.crop_size, 3])

            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458.
            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.2, upper=1.8)
        else:

            distorted_image = tf.image.resize_image_with_crop_or_pad(image,
                                                                     self.opt.hyper.crop_size, self.opt.hyper.crop_size)

        distorted_image = tf.image.resize_images(distorted_image, self.image_size)

        if standarization:
            # Subtract off the mean and divide by the variance of the pixels.
            float_image = tf.image.per_image_standardization(distorted_image)
            # float_image.set_shape([self.opt.hyper.crop_size, self.opt.hyper.crop_size, 3])
            float_image.set_shape(list(self.image_size) + [3])
        else:
            float_image = distorted_image

        return float_image
