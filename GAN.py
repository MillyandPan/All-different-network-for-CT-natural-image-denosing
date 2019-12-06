# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:58:43 2019

@author: span21
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:24 2018
@author: yeohyeongyu
"""

import tensorflow.contrib.layers as tcl
import tensorflow as tf


def generator(patch, is_training=True,  kernel_size=[3,3], output_channels=1, reuse=tf.AUTO_REUSE): #Build up the network
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(patch, 64, kernel_size,strides=(1,1), padding='same', use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), reuse=reuse)
        output = tf.compat.v1.nn.leaky_relu(output)
        shortcut1 = output
    with tf.variable_scope('block2'):
        output = tf.layers.conv2d(output, 128, kernel_size, strides=(2,2), padding='same', use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), reuse=reuse)
        output = tf.compat.v1.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training, reuse=reuse))
        shortcut2 = output
    with tf.variable_scope('block3'):
        output = tf.layers.conv2d(output, 256, kernel_size, strides=(2,2), padding='same', use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), reuse=reuse)
        output = tf.compat.v1.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training, reuse=reuse))
    for layers in range(4, 12 + 1):
        with tf.variable_scope('block4%d' % layers):
            outputtemp = output
            output = tf.layers.conv2d(output, 256, kernel_size, strides=(1,1), padding='same', name='conv1%d' % layers, use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), reuse=reuse)
            output = tf.compat.v1.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training, reuse=reuse))
            output = tf.layers.conv2d(output, 256, kernel_size, strides=(1,1), padding='same', name='conv3%d' % layers, use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), reuse=reuse)
            output = tf.layers.batch_normalization(output, training=is_training, reuse=reuse)
            output = output + outputtemp
    with tf.variable_scope('block13'):
        output = tf.layers.conv2d_transpose(output, 128, kernel_size, strides=(2,2), padding='same', use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), reuse=reuse)
        output = tf.compat.v1.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training, reuse=reuse))
        output = tf.compat.v1.concat([output,shortcut2],3)
    with tf.variable_scope('block14'):
        output = tf.layers.conv2d_transpose(output, 64, kernel_size, strides=(2,2), padding='same', use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), reuse=reuse)
        output = tf.compat.v1.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training, reuse=reuse))
        output = tf.compat.v1.concat([output,shortcut1],3)
    with tf.variable_scope('block15'):
        output = tf.layers.conv2d_transpose(output, output_channels, kernel_size,strides=(1,1), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), reuse=reuse)
        output = tf.nn.tanh(output)
    return output

def discriminator(patch, is_training=False, reuse=tf.AUTO_REUSE, output_channels=1):
    with tf.variable_scope('dblock1'):
        padded_patch = tf.pad(patch, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = tf.layers.conv2d(padded_patch, 64, 4, strides=(2,2), padding='valid', kernel_initializer=tf.random_normal_initializer(0, 0.001), reuse=reuse)
        output = tf.nn.leaky_relu(output)
    with tf.variable_scope('dblock2'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = tf.layers.conv2d(output, 128, 4, strides=(2,2), padding='valid', kernel_initializer=tf.random_normal_initializer(0, 0.001), reuse=reuse)
        output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training, reuse=reuse))
    with tf.variable_scope('dblock3'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = tf.layers.conv2d(output, 256, 4, strides=(2,2), padding='valid', kernel_initializer=tf.random_normal_initializer(0, 0.001), reuse=reuse)
        output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training, reuse=reuse))
    with tf.variable_scope('dblock14'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = tf.layers.conv2d(output, 512, 4, padding='valid', kernel_initializer=tf.random_normal_initializer(0, 0.001), reuse=reuse)
        output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training, reuse=reuse))
    with tf.variable_scope('dblock15'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = tf.layers.conv2d(output, 1, 4, padding='valid', kernel_initializer=tf.random_normal_initializer(0, 0.001), reuse=reuse)
        output = tf.nn.sigmoid(output)
    return output