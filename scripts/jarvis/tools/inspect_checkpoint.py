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
import argparse
import sys
import tensorflow as tf
from tensorflow.python.platform import app

FLAGS = None


def main(unused_argv):
    """
    Restore, Update, Save
    tested only on tesorflow 1.4
    Ref: http://t.cn/EM93dc9
    """
    if not FLAGS.checkpoint_dir:
        print("Usage: inspect_checkpoint --checkpoint_dir=checkpoint_file_directory ")
        sys.exit(1)
    else:
        tf.reset_default_graph()
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, checkpoint.model_checkpoint_path)

            trainable_variable_names = [v.name for v in tf.trainable_variables]
            # just to check all variables values
            # sess.run(tf.all_variables())
            sess.run(trainable_variable_names)
            print("tf.trainable_variables:\n")
            print(trainable_variable_names)

            all_variable_names = [v.name for v in tf.all_variables]
            sess.run(all_variable_names)
            print("tf.all_variables:\n")
            print(all_variable_names)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--checkpoint_dir",
      type=str,
      default="",
      help="Checkpoint directory. "
      "Note, the needed file is end with .meta.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)



