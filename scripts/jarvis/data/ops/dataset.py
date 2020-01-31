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

import math
import os
import tensorflow as tf
import yaml

from tensorflow.python.keras.utils import generic_utils
from jarvis.core.globals import *

__all__ = [
    'Datasets',
]


class Datasets(object):
    """A `Dataset Adapter` to load TFRecord files produced by SparkFeature."""

    def __init__(self,
                 data_paths,
                 sample_info_path,
                 group_ids=[],
                 datasets=None,
                 batch_size=DEFAULT_BATCH_SIZE,
                 epochs=DEFAULT_EPOCHS):
        """
        :param data_paths: Path of TFRecord data
        :param sample_info_path: Path of file contains sample and feature's info produced
        :param datasets:
        :param date: Date of training data
        :param group_ids: ids for gAUC
        """
        self._group_ids = group_ids
        self._batch_size = batch_size
        self._epochs = epochs
        self.data_paths = data_paths
        self.sample_info_path = sample_info_path
        self._field_info = {}
        self._features = None
        self.datasets = datasets

    def build(self,
              dataset_type='TFRecordDataset',
              shuffle=False,
              epochs=1,
              batch_size=512):
        self._parse_data_info()
        self._build_features()

        allowed_datasets = {
            'TFRecordDataset',
            'TextLineDataset'
        }
        generic_utils.validate_kwargs((dataset_type,), allowed_datasets)

        self.datasets = {k: getattr(tf.data, dataset_type)(filenames=v) for k, v in self.data_paths.items()}

        for k, v in self.data_paths.items():
            dataset = getattr(tf.data, dataset_type)(filenames=v)
            # 1. Potentially shuffle records before repeat
            if shuffle:
                dataset = dataset.shuffle(buffer_size=64 * batch_size)
            # 2. Repeat the entire dataset after shuffling
            dataset = dataset.repeat(epochs)
            # 3. Batch it up.
            dataset = dataset.map(map_func=self._parse_fn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size=batch_size)
            # 4. Prefetch buffer_size * batch_size elements
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            self.datasets[k] = dataset

    def _parse_data_info(self):
        """
        Parses training/test sample num, feature size and field info
        :return:
        """
        allowed_kwargs = {
            'sample_train',
            'sample_test',
            'fields',
        }
        if not os.path.exists(self.sample_info_path):
            raise IOError("Config file is not found: %s" % self.sample_info_path)

        with open(self.sample_info_path) as inf:
            try:
                sample_infos = yaml.load(inf)
                print("sample_infos: {}".format(sample_infos))
                # Validate optional sample infos.
                generic_utils.validate_kwargs(sample_infos.keys(), allowed_kwargs)

            except yaml.YAMLError, e:
                if hasattr(e, 'problem_mark'):
                    mark = e.problem_mark
                    error_msg = 'Error on parsing yaml file in line: {}, ' \
                                'col: {}'.format(mark.line + 1, mark.column + 1)
                else:
                    error_msg = str(e)
                raise Exception(error_msg)

        self._sample_info = sample_infos

        for info in sample_infos.get('fields', []):
            self._field_info[info['name']] = info

        print("field_infos: {}".format(self._field_info))

    def _build_features(self):
        """
        Build a `dict` mapping feature keys to `FixedLenFeature` or `VarLenFeature` or 'SparseFeature' values.
        """
        # 1. build feature structure
        features = {}

        for target in GlobalVars.target_names:
            features[target] = tf.io.FixedLenFeature([], tf.float32, 0)

        for group_id in self._group_ids:
            features[group_id] = tf.io.FixedLenFeature([], tf.int64, 0)

        for field, config in self.field_info.items():
            print("fields: {}, config: {}".format(field, config))
            if config["ftype"] == "FixedLen":
                dimension = config.get("dimension", 0)
                default_value = config.get("default", DEFAULT_VALUE[config["dtype"]])
                default_value = default_value if dimension == 0 else [default_value]*dimension
                dimension = [] if dimension == 0 else [dimension]
                features[field] = tf.io.FixedLenFeature(dimension,
                                                        getattr(tf, config["dtype"]),
                                                        default_value)
            elif config["ftype"] == "VarLen":
                features[field] = tf.io.VarLenFeature(getattr(tf, config["dtype"]))
            elif config["ftype"] == "Sparse":
                # NOTE: SparkFeature encodes feature from 1 to max_feature_num, so we add 1 here
                features[field] = tf.io.SparseFeature(index_key=field,
                                                      value_key=field + ":weight",
                                                      dtype=getattr(tf, config["dtype"].split(":")[1]),
                                                      size=config["size"]+1)
            else:
                raise ValueError("Not supported feature type: %s" % config["ftype"])

        self.features = features

    def _parse_fn(self, record):
        parsed = tf.io.parse_single_example(record, self.features)
        if len(GlobalVars.target_names) == 1:
            target = parsed[GlobalVars.target_names[0]]
        else:
            target = {}
            for target_name in GlobalVars.target_names:
                target[target_name] = parsed[target_name]

        return parsed, target

    @property
    def train_steps_per_epoch(self):
        return int(math.ceil(self.sample_info['sample_train']['row_num'] / self._batch_size))

    @property
    def train_steps(self):
        return self.train_steps_per_epoch * self._epochs

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def field_info(self):
        return self._field_info

    @property
    def fields(self):
        return self.field_info.keys()

    def field_size(self, field, add_one=True):
        """
        :param field:   field name
        :param add_one: whether add 1 to the size of field's encode space
        :return: Return the size of field's encode space
        """
        return self.field_info[field]['size'] + 1 if add_one else self.field_info[field]['size']

    @property
    def sample_info(self):
        return self._sample_info
