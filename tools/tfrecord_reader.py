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

import argparse
import tensorflow as tf


def format_output(example, columns):
    if columns is None:
        print(example)
    else:
        values = {}
        for column in columns:
            feature = example.features.feature[column]
            values.setdefault(column, [])
            kind = feature.WhichOneof('kind')
            if kind is None:
                continue
            for value in getattr(getattr(feature, kind), "value"):
                values[column].append(str(value))
        output = " ".join([key+":"+",".join(value) for key, value in values.items()])
        print(output)


def print_parsed_example(input_path, columns=None, num=0, queries=None):
    records = tf.data.TFRecordDataset([input_path])
    index = 0
    for record in records:
        if num and index >= num:
            return
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        if not queries:
            format_output(example, columns)
            index += 1
        else:
            queries_satisfied = True
            for query in queries:
                key, value = tuple(query.split(':'))
                feature = example.features.feature[key]
                kind = feature.WhichOneof('kind')
                if kind is None:
                    queries_satisfied = False
                    break
                values = getattr(getattr(feature, kind), "value")
                if len(values) == 0 or str(values[0]) != str(value):
                    queries_satisfied = False
                    break

            if queries_satisfied:
                format_output(example, columns)
                index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', help='input data path')
    parser.add_argument('--num', help='print num records', default=None)
    parser.add_argument('--features', nargs='+', type=str, help="--features target id", default=None)
    parser.add_argument('--queries', nargs='+', type=str, help="--queries id:72 target:0.0", default=None)
    args = parser.parse_args()

    print_parsed_example(args.input, columns=args.features, num=args.num, queries=args.queries)