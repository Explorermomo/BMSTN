import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils.utils import disjoint_months, infer_mask, compute_mean, geographical_distance, thresholded_gaussian_kernel
from ..utils.utils import sample_mask

class AirQuality(PandasDataset):
    SEED = 3210

    def __init__(self, impute_nans=False, small=False, freq='60T', masked_sensors=None):
        self.random = np.random.default_rng(self.SEED)
        self.test_months = [3, 6, 9, 12]
        self.infer_eval_from = 'next'
        self.eval_mask = None
        df, dist, mask = self.load(impute_nans=impute_nans, small=small, masked_sensors=masked_sensors)
        self.dist = dist
        if masked_sensors is None:
            self.masked_sensors = list()
        else:
            self.masked_sensors = list(masked_sensors)
        super().__init__(dataframe=df, u=None, mask=mask, name='air', freq=freq, aggr='nearest')

    def load_raw(self, small=False):
        if small:
            path = os.path.join(datasets_path['air'], 'small36.h5')
            eval_mask = pd.DataFrame(pd.read_hdf(path, 'eval_mask'))
        else:
            path = os.path.join(datasets_path['air'], 'full437.h5')
            eval_mask = None
        df = pd.DataFrame(pd.read_hdf(path, 'pm25'))
        stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
        return df, stations, eval_mask

    def load(self, impute_nans=True, small=False, masked_sensors=None):
        # load readings and stations metadata
        df, stations, eval_mask = self.load_raw(small)
        # compute the masks
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is not nan else 0
        if eval_mask is None:
            eval_mask = infer_mask(df, infer_from=self.infer_eval_from)

            # eval_mask = sample_mask(df.shape,
            #                         p=0.,
            #                         p_noise=0.25,
            #                         min_seq=12,  # 12
            #                         max_seq=12 * 4,  # 12 * 4
            #                         rng=np.random.default_rng(9101112))

        eval_mask = eval_mask.values.astype('uint8')
        if masked_sensors is not None:
            eval_mask[:, masked_sensors] = np.where(mask[:, masked_sensors], 1, 0)
        self.eval_mask = eval_mask  # 1 if value is ground-truth for imputation else 0
        # eventually replace nans with weekly mean by hour
        if impute_nans:
            df = df.fillna(compute_mean(df))
        # compute distances from latitude and longitude degrees
        st_coord = stations.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(st_coord, to_rad=True).values
        return df, dist, mask

    def splitter(self, dataset, val_len=1., in_sample=False, window=0):
        nontest_idxs, test_idxs = disjoint_months(dataset, months=self.test_months, synch_mode='horizon')
        if in_sample:
            train_idxs = np.arange(len(dataset))
            val_months = [(m - 1) % 12 for m in self.test_months]
            _, val_idxs = disjoint_months(dataset, months=val_months, synch_mode='horizon')
        else:
            # take equal number of samples before each month of testing
            val_len = (int(val_len * len(nontest_idxs)) if val_len < 1 else val_len) // len(self.test_months)
            # get indices of first day of each testing month
            delta_idxs = np.diff(test_idxs)
            end_month_idxs = test_idxs[1:][np.flatnonzero(delta_idxs > delta_idxs.min())]
            if len(end_month_idxs) < len(self.test_months):
                end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
            # expand month indices
            month_val_idxs = [np.arange(v_idx - val_len, v_idx) - window for v_idx in end_month_idxs]
            val_idxs = np.concatenate(month_val_idxs) % len(dataset)
            # remove overlapping indices from training set
            ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs, val_idxs, synch_mode='horizon', as_mask=True)
            train_idxs = nontest_idxs[~ovl_idxs]
        return [train_idxs, val_idxs, test_idxs]

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        theta = np.std(self.dist[:36, :36])  # use same theta for both air and air36
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        return self._mask

    @property
    def training_mask(self):
        return self._mask if self.eval_mask is None else (self._mask & (1 - self.eval_mask))

    def test_interval_mask(self, dtype=bool, squeeze=True):
        m = np.in1d(self.df.index.month, self.test_months).astype(dtype)
        if squeeze:
            return m
        return m[:, None]

class MissingValuesAirQuality(PandasDataset):
    SEED = 3210

    def __init__(self, impute_nans=False, small=False, p_fault=0.0015, p_noise=0.05, freq='60T', masked_sensors=None):
        self.random = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        self.test_months = [3, 6, 9, 12]
        self.infer_eval_from = 'next'
        self.eval_mask = None
        df, dist, mask = self.load(impute_nans=impute_nans, small=small, p_fault=self.p_fault, p_noise=self.p_noise,
                                   masked_sensors=masked_sensors)
        self.dist = dist
        if masked_sensors is None:
            self.masked_sensors = list()
        else:
            self.masked_sensors = list(masked_sensors)
        super().__init__(dataframe=df, u=None, mask=mask, name='air', freq=freq, aggr='nearest')

    def load_raw(self, small=False):
        if small:
            path = os.path.join(datasets_path['air'], 'small36.h5')
            eval_mask = None
        else:
            path = os.path.join(datasets_path['air'], 'full437.h5')
            eval_mask = None
        df = pd.DataFrame(pd.read_hdf(path, 'pm25'))
        stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
        return df, stations, eval_mask

    def load(self, impute_nans=True, small=False, p_fault=0.0015, p_noise=0.05, masked_sensors=None):
        # load readings and stations metadata
        df, stations, eval_mask = self.load_raw(small)
        # compute the masks
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is not nan else 0
        if eval_mask is None:
            eval_mask = sample_mask(df.shape,
                                    p=p_fault,
                                    p_noise=p_noise,
                                    min_seq=12,  # 12
                                    max_seq=12 * 4,  # 12 * 4
                                    rng=np.random.default_rng(9101112))
        eval_mask = eval_mask.astype('uint8')
        if masked_sensors is not None:
            eval_mask[:, masked_sensors] = np.where(mask[:, masked_sensors], 1, 0)
        self.eval_mask = eval_mask  # 1 if value is ground-truth for imputation else 0
        # eventually replace nans with weekly mean by hour
        if impute_nans:
            df = df.fillna(compute_mean(df))
        # compute distances from latitude and longitude degrees
        st_coord = stations.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(st_coord, to_rad=True).values
        return df, dist, mask

    def splitter(self, dataset, val_len=1., in_sample=False, window=0):
        nontest_idxs, test_idxs = disjoint_months(dataset, months=self.test_months, synch_mode='horizon')
        if in_sample:
            train_idxs = np.arange(len(dataset))
            val_months = [(m - 1) % 12 for m in self.test_months]
            _, val_idxs = disjoint_months(dataset, months=val_months, synch_mode='horizon')
        else:
            # take equal number of samples before each month of testing
            val_len = (int(val_len * len(nontest_idxs)) if val_len < 1 else val_len) // len(self.test_months)
            # get indices of first day of each testing month
            delta_idxs = np.diff(test_idxs)
            end_month_idxs = test_idxs[1:][np.flatnonzero(delta_idxs > delta_idxs.min())]
            if len(end_month_idxs) < len(self.test_months):
                end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
            # expand month indices
            month_val_idxs = [np.arange(v_idx - val_len, v_idx) - window for v_idx in end_month_idxs]
            val_idxs = np.concatenate(month_val_idxs) % len(dataset)
            # remove overlapping indices from training set
            ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs, val_idxs, synch_mode='horizon', as_mask=True)
            train_idxs = nontest_idxs[~ovl_idxs]
        return [train_idxs, val_idxs, test_idxs]

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        theta = np.std(self.dist[:36, :36])  # use same theta for both air and air36
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        return self._mask

    @property
    def training_mask(self):
        return self._mask if self.eval_mask is None else (self._mask & (1 - self.eval_mask))

    def test_interval_mask(self, dtype=bool, squeeze=True):
        m = np.in1d(self.df.index.month, self.test_months).astype(dtype)
        if squeeze:
            return m
        return m[:, None]


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

