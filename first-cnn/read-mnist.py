from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

#learn how data is formatted

#convert new 28x28 images into proper format for live demos