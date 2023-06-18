
#import tensorflow as tf;
import numpy as np;
from enum import Enum;
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNArchUtilsPyTorch:
#    #@staticmethod
#    #def init_weights_normal(shape, stddev):
#    #    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=stddev));

#    #@staticmethod
#    #def init_weights_constant(shape, const_val=0.1):
#    #    return tf.Variable(tf.constant(const_val, shape=shape));
    
    @staticmethod
    def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, input_width=-1, input_height=-1):
        #conv_2d = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding=padding, name=name);
        #conv_2d = tf.nn.bias_add(conv_2d, bias);

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        conv_2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias);
        output_width = -1;
        output_height = -1;
        if(input_width > 0):
            output_width = input_width - kernel_size + 1;
        if(input_height > 0):
            output_height = input_height - kernel_size + 1;
        print('conv_2d shape')
        print(conv_2d.shape)
        return conv_2d, output_width, output_height;

#    @staticmethod
#    def relu(input, name=None):
#        return tf.nn.relu(input, name=name);

#    @staticmethod
#    def sigmoid(input, name=None):
#        return tf.nn.sigmoid(input, name=name);

#    @staticmethod
#    def dropout(input, dropout_fraction, isTest, name=None):
#         return tf.layers.dropout(input, 1-dropout_fraction, training=tf.logical_not(isTest), name=name);

#    @staticmethod
#    def deconv2d(input, weights, k=2, stride=2, padding='VALID', name=None):
#        x_shape = tf.shape(input)
#        w_shape = tf.shape(weights)
#        output_shape = tf.stack([x_shape[0], x_shape[1]*k, x_shape[2]*k, w_shape[3]])
#        return tf.nn.conv2d_transpose(input, weights, output_shape, strides=[1, stride, stride, 1], padding=padding, name=name)

#    @staticmethod
#    def max_pool_2d(input, k=2, stride=2, name=None):
#        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='VALID', name=name);

    @staticmethod
    def crop_a_to_b(input_a, input_b):
        shape_a = input_a.size();
        shape_b = input_b.size();
        cropped = input_a[:, :, (shape_a[2]-shape_b[2])//2 : (shape_a[2]-shape_b[2])//2 + shape_b[2], (shape_a[3]-shape_b[3])//2 : (shape_a[3]-shape_b[3])//2 + shape_b[3]]

        return cropped;

#    @staticmethod
#    def concate_a_to_b(input_a, input_b):
#        return tf.concat([input_a, input_b], 3);

#    @staticmethod
#    def cost_cross_entropy(logits, labels, class_weights, n_classes):
#        flat_logits = tf.reshape(logits, [-1, n_classes]);
#        flat_labels = tf.reshape(labels, [-1, n_classes]);
#        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( \
#            logits=flat_logits \
#            , targets=flat_labels \
#            , pos_weight=class_weights \
#        ));

#    @staticmethod
#    def get_probability_softmax(logits):
#        return tf.nn.softmax(logits);

#    @staticmethod
#    def crop_to_shape(data, shape):
#        print('crop_to_shape');
#        print(data.shape);
#        print('shape');
#        print(shape);
#        offset_1 = (data.shape[1] - shape[1])//2;
#        offset_2 = (data.shape[2] - shape[2])//2;
#        data2 = data[:, offset_1:(-offset_1), offset_2:(-offset_2), :];
#        print('data2');
#        print(data2.shape);
        
#        return data[:, offset_1:(-offset_1), offset_2:(-offset_2), :]