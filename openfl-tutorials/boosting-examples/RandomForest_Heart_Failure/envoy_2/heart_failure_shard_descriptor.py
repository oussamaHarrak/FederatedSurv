# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import pandas as pd
import numpy as np 
import logging
from typing import List
import random

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from sklearn.model_selection import train_test_split
    

logger = logging.getLogger(__name__)

random.seed(42)

class heart_failure_ShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1, complete=False):
        """Initialize TinyImageNetDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x if complete else x[self.rank - 1::self.worldsize]
        self.y = y if complete else y[self.rank - 1::self.worldsize]

    def get_data(self):
        return self.x, self.y

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class heart_failure_ShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        x_train, x_test, y_train, y_test = self.download_data()

        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train', complete=False):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return heart_failure_ShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize,
            complete=complete
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['11']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['2']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Heart_failure dataset, shard number {self.rank}'
                    f' out of {self.worldsize}')

    def download_data(self):
        df = pd.read_csv('../heart_failure_dataset/heart_failure_clinical_records_dataset.csv' , sep = ",")
        X = df.drop(columns = ['time','DEATH_EVENT'], axis=1)
        y = df[['time' , 'DEATH_EVENT']]
        y['DEATH_EVENT'] = y['DEATH_EVENT'].astype(bool)
        x_train , x_test , y_train , y_test = train_test_split(X,y,train_size = 0.8 ,random_state = 42)
        y_test.to_csv('../heart_failure_dataset/y_test.csv', index = True)
        return x_train , x_test , y_train , y_test