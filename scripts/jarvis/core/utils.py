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

import six
from abc import ABCMeta, abstractmethod
from hdfs import InsecureClient

__all__ = [
    "Utils",
    "DEFAULT_EPOCHS",
    "DEFAULT_SHUFFLE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EVAL_BATCH_SIZE",
    "DEFAULT_EVERY_N_ITER",
    "EPSILON",
    "DEFAULT_VALUE",
    "STATISTICAL_INFO",
    "SERIALIZED_TRAIN_SAMPLE",
    "SERIALIZED_TEST_SAMPLE",
]

JARVIS_VERSION = '0.1.0'
DEFAULT_EPOCHS = 1
DEFAULT_SHUFFLE = 1
DEFAULT_BATCH_SIZE = 512
DEFAULT_EVAL_BATCH_SIZE = 1024
DEFAULT_EVERY_N_ITER = 100
EPSILON = 1e-6

STATISTICAL_INFO = "utils/statistical_info/"
SERIALIZED_TRAIN_SAMPLE = "serialized_sample/train"
SERIALIZED_TEST_SAMPLE = "serialized_sample/test"

DEFAULT_VALUE = {
    "int64": 0,
    "float32": 0.0,
    "bytes": ""
}


@six.add_metaclass(ABCMeta)
class Utils:
    # The first target name should be y_true for gAUC
    target_names = []
    eval_target_names = {}

    namenodes = ["your hdfs namenode's URI"],
    user = "your account"

    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def find_hdfs_namenode_address():

        for nd in namenodes:
            hdfs_client = InsecureClient(url=nd, user=user)
            try:
                hdfs_client.list('/')
                return nd
            except:
                continue
        raise Exception("No available name node.")
