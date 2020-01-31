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
    "accuracy",
    "auc",
    "rmse",
    "mae",
]


def accuracy(y_true, y_pred):
    y_pred = tf.cast((y_pred > 0.5), tf.int32)
    index = tf.metrics.accuracy(labels=y_true, predictions=y_pred)
    return index


def auc(y_true, y_pred):
    index = tf.metrics.auc(labels=y_true, predictions=y_pred)
    return index


def rmse(y_true, y_pred):
    return tf.metrics.root_mean_squared_error(labels=y_true, predictions=y_pred)


def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(labels=y_true, predictions=y_pred)


# TODO
def ndcg(y_true, y_pred):
    pass
