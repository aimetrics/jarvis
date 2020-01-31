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
import tensorflow as tf


def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()
    if key in registry:
        return registry[key]
    else:
        raise ValueError(
            'Key {} not supported, available options: {}'.format(
                key, registry.keys()
            )
        )


def set_default_value(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value


def gpu_info():
    infos = ["Num GPUs Available: {}".format(len(tf.config.experimental.list_physical_devices('GPU')))]
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                infos.append('gpu:{}'.format(gpu))
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            infos.append('{} Physical GPUs, {} Logical GPUs'.format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    return "\n".join(infos)
