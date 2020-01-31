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
import csv
import logging
import multiprocessing
import pickle
import shutil
import sys
import tensorflow as tf
import yaml
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from tensorflow.python.platform import gfile
from tqdm import tqdm
from yaml.representer import Representer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AvazuDataset(object):
    """
    Avazu Click-Through Rate Prediction Dataset

    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature

    :param dataset_path: avazu train path
    :param min_threshold: infrequent feature threshold

    Reference
        https://www.kaggle.com/c/avazu-ctr-prediction
    """

    def __init__(self, dataset_path=None, feature_map_cache='.feature_map',
                 rebuild_feature_map=False, min_threshold=4):
        self.path = dataset_path
        self.train_cnt = 0
        self.test_cnt = 0
        self.feature_map_cache = feature_map_cache
        self.rebuild_feature_map = rebuild_feature_map
        self.min_threshold = min_threshold
        self.field_names = None
        self.target_name = None
        self.field_info = None
        self.idx_to_field_name = None
        self.feature_map = None

        self._build()

    def _build(self):
        self._get_field_name()
        self._get_feature_map()

    def _get_field_name(self):
        with gfile.Open(self.path) as csv_file:  # open the input file.
            data_file = csv.reader(csv_file)
            header = next(data_file)  # get the header line.
            self.field_info = {k: v for v, k in enumerate(header)}
            self.idx_to_field_name = {idx: name for idx, name in enumerate(header)}
            self.field_names = header[2:]  # list of feature names.
            self.field_names.append(header[0])
            self.target_name = header[1]  # target name.

    # The following functions can be used to convert a value to a type compatible with tf.Example.
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _get_feature_map(self):
        if not self.rebuild_feature_map and Path(self.feature_map_cache).exists():
            with open(self.feature_map_cache, 'rb') as f:
                feature_mapper = pickle.load(f)
        else:
            feature_cnts = defaultdict(lambda: defaultdict(int))
            with open(self.path) as f:
                f.readline()
                pbar = tqdm(f, mininterval=1, smoothing=0.1)
                pbar.set_description('Create avazu dataset: counting features')
                for line in pbar:
                    values = line.rstrip('\n').split(',')
                    if len(values) != len(self.field_names) + 1:
                        continue
                    for k, v in self.field_info.items():
                        if k not in ['click']:
                            feature_cnts[k][values[v]] += 1
            feature_mapper = {field_name: {feature_name for feature_name, c in cnt.items() if c >= self.min_threshold}
                              for field_name, cnt in feature_cnts.items()}
            feature_mapper['id'] = {feature_name for feature_name, c in feature_cnts['id'].items()}
            feature_mapper = {field_name: {feature_name: idx for idx, feature_name in enumerate(cnt)}
                              for field_name, cnt in feature_mapper.items()}

            shutil.rmtree(self.feature_map_cache, ignore_errors=True)
            with open(self.feature_map_cache, 'wb') as f:
                pickle.dump(feature_mapper, f)

        self.feature_map = feature_mapper

    def to_config(self, path):
        data_info = defaultdict(list)
        for field, feature_index in self.feature_map.items():
            field_info = defaultdict()
            field_info['name'] = field
            field_info['size'] = len(feature_index)
            field_info['ftype'] = 'Sparse'
            field_info['dtype'] = 'int64:float32'

            data_info['fields'].append(field_info)

        if self.train_cnt < 1:
            raise ValueError("train sample cnt less then 1")
        if self.test_cnt < 1:
            raise ValueError("test sample cnt less then 1")
        data_info['sample_train'] = {'row_num': self.train_cnt}
        data_info['sample_test'] = {'row_num': self.test_cnt}

        with open(path, 'w') as f:
            yaml.add_representer(defaultdict, Representer.represent_dict)
            yaml.dump(data_info, f)

    def convert2tfrecord(self, output_path=None, data_type="train"):
        """Transforms avazu data in text format to tf.Example protos and dump to a TFRecord file.
        The benefit of doing this is to use existing training and evaluating functionality within tf
        packages.

        NOTE. Currently we don't support or use tf.SequenceExample, only tf.Example is supported.
        Args:
            output_path: string, path to the TFRecord file for transformed tf.Example protos.
        """
        feature_mapper = self.feature_map

        def parsing_loop(in_queue=None, out_queue=None):
            """
            function to be executed within each parsing process.

            Args:
              in_queue: the queue used to store avazu data records as strings.
              out_queue: the queue used to store serialized tf.Examples as strings.
            """
            while True:  # loop.
                raw_record = in_queue.get()  # read from in_queue.
                logging.debug('parsing_loop raw_example:{}'.format(raw_record))
                if raw_record == "DONE":
                    # We were done here.
                    break
                features = {}  # dict for all feature columns and target column.
                # parse the record according to proto definitions.
                values = raw_record.rstrip('\n').split(',')
                if len(values) != len(self.field_names) + 1:
                    continue
                features = {self.idx_to_field_name[idx]: self._int64_feature(feature_mapper[self.idx_to_field_name[idx]][value]) for idx, value in enumerate(values)
                            if self.idx_to_field_name[idx] != 'click' and value in feature_mapper[self.idx_to_field_name[idx]]}
                feature_values = {self.idx_to_field_name[idx]+':weight': self._float_feature(1) for idx, value in enumerate(values)
                                  if self.idx_to_field_name[idx] != 'click' and value in feature_mapper[self.idx_to_field_name[idx]]}

                features.update(feature_values)
                features.update({'target': self._float_feature(float(values[1]))})
                logging.debug('parsing_loop features:{}'.format(features))
                logging.debug('parsing_loop feature_values:{}'.format(feature_values))

                # create an instance of tf.Example.
                example = tf.train.Example(features=tf.train.Features(feature=features))
                # serialize the tf.Example to string.
                raw_example = example.SerializeToString()

                # write the serialized tf.Example out.
                out_queue.put(raw_example)

        def writing_loop(out_queue, out_file):
            """
            function to be executed within the single writing process.

            Args:
              out_queue: the queue used to store serialized tf.Examples as strings.
              out_file: string, path to the TFRecord file for transformed tf.Example protos.
            """
            writer = tf.io.TFRecordWriter(out_file)  # writer for the output TFRecord file.
            sample_count = 0
            while True:
                raw_example = out_queue.get()  # read from out_queue.
                logging.debug('writing_loop raw_example:{}'.format(raw_example))
                if raw_example == "DONE":
                    break
                writer.write(raw_example)  # write it out.
                sample_count += 1
                if not sample_count % 1000:
                    logging.info('%s Processed %d examples' % (datetime.now(), sample_count))
                    sys.stdout.flush()
            writer.close()  # close the writer.
            logging.info('%s >>>> Processed %d examples <<<<' % (datetime.now(), sample_count))
            self.sample_cnt = sample_count
            sys.stdout.flush()

        in_queue = Queue()  # queue for raw gdt training data records.
        out_queue = Queue()  # queue for serialized tf.Examples.
        # start parsing processes.
        num_parsers = int(multiprocessing.cpu_count() - 2)
        parsers = []
        for i in range(num_parsers):
            p = Process(target=parsing_loop, args=(in_queue, out_queue))
            parsers.append(p)
            p.start()

        # start writing process.
        writer = Process(target=writing_loop, args=(out_queue, output_path))
        writer.start()
        logging.info('%s >>>> BEGIN to feed input file %s <<<<' % (datetime.now(), self.path))
        # read a record in.
        with open(self.path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('reading avazu dataset')
            line_num = 0
            train_cnt = 0
            test_cnt = 0
            for line in pbar:
                if line_num == 0:
                    line_num += 1
                    continue
                if data_type == "train":
                    if "141030" in line.rstrip('\n').split(',')[2]:
                        test_cnt += 1
                        continue
                    train_cnt += 1
                else:
                    if "141030" not in line.rstrip('\n').split(',')[2]:
                        continue
                in_queue.put(line)  # write to in_queue.
            self.train_cnt = train_cnt
            self.test_cnt = test_cnt
        # terminate and wait for all parsing processes.
        for i in range(num_parsers):
            in_queue.put("DONE")
        for i in range(num_parsers):
            parsers[i].join()

        # terminate and wait for the writing process.
        out_queue.put("DONE")
        writer.join()
        logging.info('%s >>>> END of consuming input file %s <<<<' % (datetime.now(), self.path))
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', help='input data path')
    args = parser.parse_args()
    dataset = AvazuDataset(dataset_path=args.input)
    dataset.convert2tfrecord(output_path=args.input + '_train.tfrecord', data_type='train')
    dataset.to_config(path=args.input + '.yaml')
    dataset.convert2tfrecord(output_path=args.input + '_test.tfrecord', data_type='test')
