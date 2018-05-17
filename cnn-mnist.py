from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(images, labels, mode):
    #CNN logic

    #Input Layer
    input_layer = tf.reshape(images["x"],[-1,28,28,-1])

    #1 Convolutional Layer: 32 5x5 filters with ReLU activation
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu
    )

    #1 Pooling Layer: Max pooling with 2x2 filter (no overlap)
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = [2,2],
        strides = 2 #like padding
    )

    #2 Covolutional Layer: 64 5x5 filters with ReLU activation
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu
    )

    #2 Pooling Layer: same as #1
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = [2,2],
        strides = 2
    )

    #1 Dense Layer: 1,024 neurons with regularization rate of 0.4
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(
        inputs = pool2_flat,
        units = 1024,
        activation = tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    #2 Dense Layer (Logits): 10 neurons (0-9) "Output Layer"
    logits = tf.layers.dense(inputs = dropout, units = 10)

    #Training & Evaluation
    predictions = {
        #generate some predictions
        "classes": tf.argmax(inputs = logits, axis = 1),
        
        #add `softmax_tensor` to graph to help with PREDICT
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

if __name__ == "__main__":
    tf.app.run()