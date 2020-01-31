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
# pylint: disable=g-import-not-at-top
"""Callbacks: utilities called at certain points during model training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import tensorflow as tf


class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 interval=100):
        self.interval = interval
        self.start = datetime.datetime.utcnow()
        self.end = None
        super(TimeCallback, self).__init__()

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.interval == 0:
            self.end = datetime.datetime.utcnow()
            print(' - step = {} ({} secs)'.format(batch, (self.end - self.start).total_seconds()))
            self.start = datetime.datetime.utcnow()
