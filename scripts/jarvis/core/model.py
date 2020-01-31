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

import abc
import six

import tensorflow as tf
from tensorflow.python.keras import optimizers


@six.add_metaclass(abc.ABCMeta)
class Model(object):
    """`Model` groups layers into an object with training and inference features.

    By subclassing the `Model` class: in that case, you should define your
    layers in `__init__` and you should implement the model's forward pass
    in `call`.

    ```python
    import tensorflow as tf

    class MyModel(Model):

        def __init__(self):
          super(MyModel, self).__init__()
          self.fc_64 = jarvis.layers.FullyConnect(units=64, activation=tf.nn.relu, input_shape=(input_feature_size, ))
          self.fc_1 = jarvis.layers.FullyConnect(units=1)

        def call(self, inputs):
          x = self.fc_64(inputs)
          x = self.fc_1(x)
          logit = tf.sigmoid(x)
          return logit

    model = MyModel()
    ```
    """

    def __init__(self,
                 optimizer,
                 loss=None,
                 metrics=None,
                 loss_weights=None,
                 sample_weight_mode=None,
                 weighted_metrics=None,
                 target_tensors=None,
                 **kwargs):
        """Configures the model for training.

        Arguments:
            optimizer: String (name of optimizer) or optimizer instance.
                See `tf.keras.optimizers`.
            loss: String (name of objective function), objective function or
                `tf.losses.Loss` instance. See `tf.losses`. If the model has
                multiple outputs, you can use a different loss on each output by
                passing a dictionary or a list of losses. The loss value that will
                be minimized by the model will then be the sum of all individual
                losses.
            metrics: List of metrics to be evaluated by the model during training
                and testing. Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary, such as
                `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
                You can also pass a list (len = len(outputs)) of lists of metrics
                such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
                `metrics=['accuracy', ['accuracy', 'mse']]`.
            loss_weights: Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions
                of different model outputs.
                The loss value that will be minimized by the model
                will then be the *weighted sum* of all individual losses,
                weighted by the `loss_weights` coefficients.
                If a list, it is expected to have a 1:1 mapping
                to the model's outputs. If a tensor, it is expected to map
                output names (strings) to scalar coefficients.
            sample_weight_mode: If you need to do timestep-wise
                sample weighting (2D weights), set this to `"temporal"`.
                `None` defaults to sample-wise weights (1D).
                If the model has multiple outputs, you can use a different
                `sample_weight_mode` on each output by passing a
                dictionary or a list of modes.
            weighted_metrics: List of metrics to be evaluated and weighted
                by sample_weight or class_weight during training and testing.
            target_tensors: By default, Keras will create placeholders for the
                model's target, which will be fed with the target data during
                training. If instead you would like to use your own
                target tensors (in turn, Keras will not expect external
                Numpy data for these targets at training time), you
                can specify them via the `target_tensors` argument. It can be
                a single tensor (for a single-output model), a list of tensors,
                or a dict mapping output names to target tensors.
                **kwargs: Any additional arguments.

        Raises:
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`, or `metrics`.
        """
        self._validate_kwargs(kwargs, {'trainable, last_dim'},
                              'Functional models may only specify `name` and `trainable` keyword '
                              'arguments during initialization. Got an unexpected argument:')

        self._inputs = None
        self._outputs = None
        self._model = None

        self._optimizer = optimizers.get(optimizer)
        self._loss = loss or {}
        self._metrics = metrics or {}
        self._loss_weights = loss_weights
        self._sample_weight_mode = sample_weight_mode
        self._metrics = metrics or []
        self._weighted_metrics = weighted_metrics
        self._target_tensors = target_tensors
        self._kwargs = kwargs

        self.built = False
        self.compiled = False

    @staticmethod
    def _validate_kwargs(kwargs, allowed_kwargs,
                         error_message='Keyword argument not understood:'):
        """Checks that all keyword arguments are in the set of allowed keys."""
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError(error_message, kwarg)

    def build(self):
        """Creates the layers needed for a computational graph.

        This is a method that implementers of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.

        """
        self.built = True
        self.model = tf.keras.Model(self.inputs, self.outputs)

    def compile(self, **kwargs):
        """Configures the model for training."""
        if self.built:
            self.model.compile(optimizer=self._optimizer,
                               loss=self._loss,
                               metrics=self._metrics,
                               loss_weights=self._loss_weights,
                               sample_weight_mode=self._sample_weight_mode,
                               weighted_metrics=self._weighted_metrics,
                               target_tensors=self._target_tensors,
                               **kwargs)

            self.compiled = True

        else:
            raise ValueError('Model is not built yet.')

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Arguments:
            x: Input data. It could be:
              - A `tf.data` dataset or a dataset iterator. Should return a tuple
                of either `(inputs, targets)` or `(inputs, targets, sample_weights)`.
              - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
              - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
              - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
              - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
                or `(inputs, targets, sample weights)`.
            y: Target data. Like the input data `x`,
              it could be either Numpy array(s) or TensorFlow tensor(s).
              It should be consistent with `x` (you cannot have Numpy inputs and
              tensor targets, or inversely). If `x` is a dataset, dataset
              iterator, generator, or `keras.utils.Sequence` instance, `y` should
              not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of symbolic tensors, dataset, dataset iterators,
                generators, or `keras.utils.Sequence` instances (since they generate
                batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                Note that the progress bar is not particularly useful when
                logged to a file, so verbose=2 is recommended when not running
                interactively (eg, in a production environment).
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset, dataset iterator, generator or
               `keras.utils.Sequence` instance.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                  - tuple `(x_val, y_val)` of Numpy arrays or tensors
                  - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
                  - dataset or a dataset iterator
                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` must be provided.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a float (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`. This argument is not
                supported when `x` is a dataset, dataset iterator, generator, or
               `keras.utils.Sequence` instance, instead provide the sample_weights
                as the third element of `x`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If x is a
                `tf.data` dataset or a dataset iterator, and 'steps_per_epoch'
                is None, the epoch will run until the input dataset is exhausted.
            validation_steps: Only relevant if `validation_data` is provided and
                is a dataset or dataset iterator. Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If validation_data is a `tf.data` dataset
                or a dataset iterator, and 'validation_steps' is None, validation
                will run until the `validation_data` dataset is exhausted.
            validation_freq: Only relevant if validation data is provided. Integer
                or `collections.Container` instance (e.g. list, tuple, etc.). If an
                integer, specifies how many training epochs to run before a new
                validation run is performed, e.g. `validation_freq=2` runs
                validation every 2 epochs. If a Container, specifies the epochs on
                which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
                validation at the end of the 1st, 2nd, and 10th epochs.
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1. If 0, will execute the generator on the main
                thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
            **kwargs: Used for backwards compatibility.

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        Raises:
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        if self.compiled:
            history = self.model.fit(x=x,
                                     y=y,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     verbose=verbose,
                                     callbacks=callbacks,
                                     validation_split=validation_split,
                                     validation_data=validation_data,
                                     shuffle=shuffle,
                                     class_weight=class_weight,
                                     sample_weight=sample_weight,
                                     initial_epoch=initial_epoch,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_steps=validation_steps,
                                     validation_freq=validation_freq,
                                     max_queue_size=max_queue_size,
                                     workers=workers,
                                     use_multiprocessing=use_multiprocessing,
                                     **kwargs)
            return history
        else:
            raise ValueError('Model is not compiled yet.')

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False):
        """Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches.

        Arguments:
            x: Input data. It could be:
              - A `tf.data` dataset or a dataset iterator.
              - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
              - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
              - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
              - A generator or `keras.utils.Sequence` instance.
            y: Target data. Like the input data `x`,
              it could be either Numpy array(s) or TensorFlow tensor(s).
              It should be consistent with `x` (you cannot have Numpy inputs and
              tensor targets, or inversely).
              If `x` is a dataset, dataset iterator, generator or
              `keras.utils.Sequence` instance, `y` should not be specified (since
              targets will be obtained from the iterator/dataset).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` is your data is in the
                form of symbolic tensors, dataset, dataset iterators,
                generators, or `keras.utils.Sequence` instances (since they generate
                batches).
            verbose: 0 or 1. Verbosity mode.
                0 = silent, 1 = progress bar.
            sample_weight: Optional Numpy array of weights for
                the test samples, used for weighting the loss function.
                You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`. This argument is not
                supported when `x` is a dataset or a dataset iterator, instead pass
                sample weights as the third element of `x`.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.
                If x is a `tf.data` dataset or a dataset iterator, and `steps` is
                None, 'evaluate' will run until the dataset is exhausted.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
                See [callbacks](/api_docs/python/tf/keras/callbacks).
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up when using
                process-based threading. If unspecified, `workers` will default
                to 1. If 0, will execute the generator on the main thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.

        Returns:
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        Raises:
            ValueError: in case of invalid arguments.
        """
        return self.model.evaluate(x=x,
                                   y=y,
                                   batch_size=batch_size,
                                   verbose=verbose,
                                   sample_weight=sample_weight,
                                   steps=steps,
                                   callbacks=callbacks,
                                   max_queue_size=max_queue_size,
                                   workers=workers,
                                   use_multiprocessing=use_multiprocessing)

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        Arguments:
            x: Input samples. It could be:
              - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
              - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
              - A `tf.data` dataset or a dataset iterator.
              - A generator or `keras.utils.Sequence` instance.
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` is your data is in the
                form of symbolic tensors, dataset, dataset iterators,
                generators, or `keras.utils.Sequence` instances (since they generate
                batches).
            verbose: Verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`. If x is a `tf.data`
                dataset or a dataset iterator, and `steps` is None, `predict` will
                run until the input dataset is exhausted.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.
                See [callbacks](/api_docs/python/tf/keras/callbacks).
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up when using
                process-based threading. If unspecified, `workers` will default
                to 1. If 0, will execute the generator on the main thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.


        Returns:
            Numpy array(s) of predictions.

        Raises:
            ValueError: In case of mismatch between the provided
                input data and the model's expectations,
                or in case a stateful model receives a number of samples
                that is not a multiple of the batch size.
        """

        return self.model.predict(x=x,
                                  batch_size=batch_size,
                                  verbose=verbose,
                                  steps=steps,
                                  callbacks=callbacks,
                                  max_queue_size=max_queue_size,
                                  workers=workers,
                                  use_multiprocessing=use_multiprocessing)

    def summary(self, line_length=None, positions=None, print_fn=None):
        """Prints a string summary of the network.

        Arguments:
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided,
                defaults to `[.33, .55, .67, 1.]`.
            print_fn: Print function to use. Defaults to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.

        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        return self.model.summary(line_length=line_length, positions=positions, print_fn=print_fn)

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None):
        """Saves the model to Tensorflow SavedModel or a single HDF5 file.

        The savefile includes:
            - The model architecture, allowing to re-instantiate the model.
            - The model weights.
            - The state of the optimizer, allowing to resume training
                exactly where you left off.

        This allows you to save the entirety of the state of a model
        in a single file.

        Saved models can be reinstantiated via `keras.models.load_model`.
        The model returned by `load_model`
        is a compiled model ready to be used (unless the saved model
        was never compiled in the first place).

        Arguments:
            filepath: String, path to SavedModel or H5 file to save the model.
            overwrite: Whether to silently overwrite any existing file at the
                target location, or provide the user with a manual prompt.
            include_optimizer: If True, save optimizer's state together.
            save_format: Either 'tf' or 'h5', indicating whether to save the model
              to Tensorflow SavedModel or HDF5. The default is currently 'h5', but
              will switch to 'tf' in TensorFlow 2.0. The 'tf' option is currently
              disabled (use `tf.keras.experimental.export_saved_model` instead).

        Example:

        ```python
        from keras.models import load_model

        model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
        del model  # deletes the existing model

        # returns a compiled model
        # identical to the previous one
        model = load_model('my_model.h5')
        ```
        """
        self.model.save(filepath=filepath, overwrite=overwrite,
                        include_optimizer=include_optimizer, save_format=save_format)

    def get_config(self):
        return self.model.get_config()

    def to_json(self, **kwargs):
        """Returns a JSON string containing the network configuration.

        To load a network from a JSON save file, use
        `keras.models.model_from_json(json_string, custom_objects={})`.

        Arguments:
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        Returns:
            A JSON string.
        """
        return self.model.to_json(**kwargs)

    def get_weights(self):
        """Retrieves the weights of the model.

        Returns:
            A flat list of Numpy arrays.
        """
        return self.model.get_weights()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value


