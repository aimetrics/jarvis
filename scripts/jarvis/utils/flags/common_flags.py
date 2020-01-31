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

from absl import flags
from jarvis.core.globals import DEFAULT_EVERY_N_ITER, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE


def define_flags():
    flags.DEFINE_integer('log_interval', DEFAULT_EVERY_N_ITER, 'batches between logging training status.')
    flags.DEFINE_integer('cuda_device', 0, 'GPU device num, default 0.')
    flags.DEFINE_string('date', '', 'train date.')
    flags.DEFINE_string('train_data_path', '', 'The location of the training data.')
    flags.DEFINE_string('eval_data_path', '', 'The location of the evaluation data.')
    flags.DEFINE_string('sample_info_path', '', 'The location of the data information')
    flags.DEFINE_boolean('debug', False, '')
    flags.DEFINE_integer('epochs', DEFAULT_EPOCHS, 'The number of epochs used to train.')
    flags.DEFINE_boolean('shuffle', False, 'shuffle dataset or not')
    flags.DEFINE_integer('batch_size', DEFAULT_BATCH_SIZE, 'batch size for training.')
    flags.DEFINE_integer('eval_batch_size', DEFAULT_EVAL_BATCH_SIZE, 'batch size for evaluation.')
    flags.DEFINE_string('model_dir', 'model', 'The location of the model checkpoint.')
    flags.DEFINE_string('export_dir', 'serving', 'The location of a SavedModel serialization of the model.')
    flags.DEFINE_string('logs_dir', 'logs', 'The location of a SavedModel serialization of the model.')


def set_defaults(**kwargs):
    for key, value in kwargs.items():
        flags.FLAGS.set_default(key, value=value)
