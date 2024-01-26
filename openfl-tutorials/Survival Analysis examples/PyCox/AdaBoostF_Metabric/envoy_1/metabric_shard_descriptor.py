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
from openfl.utilities.data_splitters.numpy import DirichletNumPyDataSplitter
from openfl.utilities.data_splitters.numpy import LogNormalNumPyDataSplitter
from openfl.utilities.data_splitters.numpy import QuantitySkewLabelsSplitter
from openfl.utilities.data_splitters.numpy import RandomNumPyDataSplitter

from sklearn.model_selection import train_test_split

from pycox.datasets import metabric

from sklearn.compose import make_column_selector
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF, ColumnTransformerDF, SimpleImputerDF, StandardScalerDF
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
logger = logging.getLogger(__name__)

random.seed(42)

class metabric_ShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1, complete=False):
        """Initialize TinyImageNetDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        splitter =  QuantitySkewLabelsSplitter(class_per_client = 2)
        event_indicator = y.iloc[:,1].values
        idx = splitter.split(event_indicator , self.worldsize)[self.rank - 1]       
        self.x = x if complete else x[idx , :]
        self.y = y if complete else  y.iloc[idx,:] 
        #self.x = x if complete else #x[self.rank - 1::self.worldsize]
        #self.y = y if complete else #y[self.rank - 1::self.worldsize]
        

    def get_data(self):
        return self.x, self.y

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class metabric_ShardDescriptor(ShardDescriptor):
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
        return metabric_ShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize,
            complete=complete
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['10']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['2']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'metabric dataset, shard number {self.rank}'
                    f' out of {self.worldsize}')
    def _get_preprocess_transformer(self) -> ColumnTransformerDF:
        sel_fac = make_column_selector(pattern='^fac\\_')
        enc_fac = PipelineDF(steps=[('ohe', OneHotEncoderDF(sparse=False, drop='if_binary', handle_unknown='ignore'))])
        sel_num = make_column_selector(pattern='^num\\_')
        enc_num = PipelineDF(steps=[('impute', SimpleImputerDF(strategy='median')), ('scale', StandardScalerDF())])
        tr = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_fac), ('s', enc_num, sel_num)])
        return tr
    def _split_dataframe(self , df: pd.DataFrame, split_size=0.2) :
        return train_test_split(df, stratify=df['event'], test_size=split_size)
    def download_data(self):
        df_train = metabric.read_df()
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]

        x_mapper = DataFrameMapper(standardize + leave)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')
        get_target = lambda df: np.array([df['duration'].values, df['event'].values])
        y_train = get_target(df_train)
        y_test = get_target(df_test)
        y_train = pd.DataFrame(data=y_train.T, columns=['duration', 'event'])
        y_test = pd.DataFrame(data=y_test.T, columns=['duration', 'event'])
        return x_train , x_test , y_train , y_test












