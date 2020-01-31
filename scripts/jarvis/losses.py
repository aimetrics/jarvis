# -*- coding:utf-8 -*-
# Copyright 2019 The Jarvis Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = [
    "sigmoid_cross_entropy_with_class_weighting",
    "binary_crossentropy"
]


def binary_crossentropy(y_true, y_pred):
    with tf.name_scope("BinaryCrossEntropy"):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,
                                                    labels=y_true))


def sigmoid_cross_entropy_with_class_weighting(logits,
                                               labels,
                                               class_weights):
    with tf.name_scope("WeightedCrossEntropy"):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                logits=logits,
                                                                name="sigmoid_cross_entropy_per_example")
        weighted_cross_entropy = tf.multiply(class_weights, cross_entropy)

        loss = tf.reduce_mean(weighted_cross_entropy, name="sigmoid_cross_entropy")

        return loss

