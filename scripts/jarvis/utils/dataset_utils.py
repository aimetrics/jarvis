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

import os
from hdfs import InsecureClient
from jarvis.core.globals import GlobalVars


def list_files(data_source):
    if isinstance(data_source, list):
        data_files = [data_file for data_file in data_source
                      if os.path.isfile(data_file) or data_file.startswith("hdfs://")]
        return data_files

    # tensorflow run on hdfs:
    # https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/hadoop.md
    if data_source.startswith("hdfs://"):
        data_filenames = list(data_source)
        return data_filenames

    if not os.path.exists(data_source):
        raise ValueError('data file or dir not exists')
    if os.path.isfile(data_source):
        data_files = [data_source]
    elif os.path.isdir(data_source):
        data_files = os.listdir(data_source)
        data_files = [os.path.join(data_source, f) for f in data_files]
    else:
        raise ValueError('Invalid data source')

    return data_files


def list(path):
    hdfs_client = InsecureClient(GlobalVars.find_hdfs_namenode_address(),
                                 user=GlobalVars.user)

    path += '/'
    relative_path = path
    if path.startswith('hdfs://'):
        pos = path.find('/', 7)
        relative_path = path[pos:]

    files = hdfs_client.list(relative_path)
    return [path + filename for filename in files if filename.startswith("part-")]
