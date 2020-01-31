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

from jarvis.utils.misc import get_from_registry

__all__ = [
    "get_initializer",
]

initializers_registry = {
    'constant': tf.initializers.constant,
    'glorot_normal': tf.initializers.glorot_normal,
    'glorot_uniform': tf.initializers.glorot_uniform,
    None: tf.initializers.glorot_normal,
}


def get_initializer(parameters):
    if parameters is None:
        return initializers_registry[parameters]()
    elif isinstance(parameters, str):
        initializer_fun = get_from_registry(
            parameters, initializers_registry)
        return initializer_fun()
    elif isinstance(parameters, dict):
        initializer_fun = get_from_registry(
            parameters['type'], initializers_registry)
        arguments = parameters.copy()
        del arguments['type']
        return initializer_fun(**arguments)
    else:
        raise ValueError(
            'Initializers parameters should be either strings or dictionaries, '
            'but the provided parameters are a {}. '
            'Parameters values: {}'.format(
                type(parameters), parameters
            ))
