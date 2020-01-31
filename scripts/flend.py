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

from absl import app
from absl import flags
import os
import tensorflow as tf

from jarvis.callbacks import TimeCallback
from jarvis.core.globals import *
from jarvis.data.ops.dataset import Datasets
from jarvis.models.flend import FLEND
from jarvis.utils.misc import gpu_info
from jarvis.utils.flags import common_flags


def main(_):
    flags_obj = flags.FLAGS
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags_obj.cuda_device)
    print(gpu_info())
    group_ids = ["user"]
    GlobalVars.target_names.append('target')

    # 1. Fetch tfrecords dataset
    data_paths = {'train': flags_obj.train_data_path,
                  'eval': flags_obj.eval_data_path}

    datasets = Datasets(data_paths=data_paths,
                        sample_info_path=flags_obj.sample_info_path,
                        group_ids=group_ids)
    datasets.build(dataset_type='TFRecordDataset')

    train_dataset = datasets.datasets['train']

    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.04, initial_accumulator_value=1e-3)
    metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryCrossentropy()]

    fields = datasets.fields
    field_info = {field: datasets.field_size(field=field, add_one=True) for field in fields}

    model = FLEND(field_info=field_info)
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    callbacks = [
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir=flags_obj.logs_dir, histogram_freq=1, profile_batch=3),
        tf.keras.callbacks.ModelCheckpoint(filepath=flags_obj.model_dir),
        TimeCallback()
    ]

    steps_per_epoch = None if not flags_obj.debug else 4
    steps = None if not flags_obj.debug else 4
    history = model.fit(train_dataset, callbacks=callbacks, steps_per_epoch=steps_per_epoch)
    print('\nhistory dict:', history.history)
    eval_dataset = datasets.datasets['eval']
    rs = model.evaluate(eval_dataset, steps=steps)
    print('\nevaluate result:', rs)


if __name__ == "__main__":
    common_flags.define_flags()
    flags.adopt_module_key_flags(common_flags)

    data_root = '/data1/cwq1/data/avazu/tiny_groups/'
    date = '20141029'
    common_flags.set_defaults(
        date=date,
        train_data_path=data_root + 'serialized_sample/train/' + date,
        eval_data_path=data_root + 'serialized_sample/test/' + date,
        sample_info_path=data_root + 'utils/statistical_info/' + date,
        model_dir='model_flend',
        logs_dir='logs_flend',
        cuda_device=1,
    )
    app.run(main)

