# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gin
import numpy as numpy
import tensorflow as tf

from google.cloud import bigtable

from tf_agents.replay_buffers import replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from tensorflow.python.data.util import nest as data_nest


@gin.configurable
class BigtableReplayBuffer(replay_buffer.ReplayBuffer):
  def __init__(self, data_spec, batch_size, opening, max_length=1000, scope='BigtableBuffer',
      project_id=None, instance_id=None, table_id=None):
    self._batch_size = batch_size
    self._max_length = max_length
    self._data_spec = data_spec
    self._opening = opening

    capacity = self._batch_size * self._max_length
    super(BigtableReplayBuffer, super).__init__(data_spec, capacity)

    if opening not in ['writing', 'reading']:
      raise ValueError('opening must be either [writing, reading]')

    if project_id is None:
      raise ValueError("BigtableRepalyBuffer requires a GCP Project ID")
    self._project_id = project_id
    if instance_id is None:
      raise ValueError("BigtableReplayBuffer reqires a Bigtable instance ID")
    self._instance_id = instance_id
    if table_id is None:
      self._table_id = scope

    def _create_unique_slot_name(spec):
      return tf.compat.v1.get_default_graph().unique_name(spec.name or 'slot')

    self._slots = tf.nest.map_structure(_create_unique_slot_name,
                                        self._tensor_spec)
    if opening == 'writing':
      print('Creating the {} table'.format(table_id))

      client = bigtable.Client(project=project_id)
      instance = client.instance(instance_id)
      self._base_table = instance.table(table_id)

      if self._base_table.exists():
        raise RuntimeError("Bigtable table with {} already exists".format(table_id))
      column_names = tf.nest.flatten(self._slots)
      max_versions_rules = [bigtable.column_family.MaxVersionsGCRule(1) 
          for _  in column_names]

      columns = dict(zip(column_names, max_versions_rules))
      self._base_table.create(column_families=columns)

    client = tf.contrib.bigtable.BigtableClient(project_id, instance_id)
    self._table = client.table(table_id)

  def serialize_to_dataset(self, items):
    pass
  
  def _add_batch(self, items):
    self.nest.assert_same_structure(items, self._data_spec)
    with tf.device('/cpu:0'), tf.name_scope(self._scope):
      serialized_data = self.serialize_to_dataset(items)
      self._table(serialized_data, 
                  COLUMN_FAMILY, 
                  COLUMNS)

  def _get_next(self, sample_batch_size=None, num_steps=None, time_stacked=True):
    pass

  def _as_dataset(self, sample_batch_size, num_steps=None, num_parallel_calls=None):
    pass

  def _gather_all(self):
    pass

  def _clear(self):
    self._base_table.drop_by_prefix('')

  def shutdown(self):
    self._base_table.delete()