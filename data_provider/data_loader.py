import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features,convert_timestamp_to_int
import warnings
from bisect import bisect
import configparser
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train',
                 features='S', data_path='weather.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.args = args
        # info
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        # init

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.config = configparser.ConfigParser()
        self.config.read(args.config_path)
        self.endogenous_list = self.config.get('Endogenous', 'target').split(',')
        self.exogenous_list = self.config.get('Exogenous', 'target').split(',')
        self.patch_len = args.patch_len
        self.scaler_endogenous = StandardScaler()
        self.scaler_exogenous = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)  # 这里面没有考虑pred的那部分，相当于num_train就是单纯的sample的个数
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0,num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]#从第一个值是num_train可以看到没有考虑pred的部分，这部分落在了vali里面，感觉验证集合训练集还是重合了
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        endogenous_data = df_raw[self.endogenous_list]
        exogenous_data = df_raw[self.exogenous_list]

        if self.scale:
            endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
            exogenous_train_data = exogenous_data[border1s[0]:border2s[0]]
            self.scaler_endogenous.fit(endogenous_train_data.values)
            self.scaler_exogenous.fit(exogenous_train_data.values)
            endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
            exogenous_data = self.scaler_exogenous.transform(exogenous_data.values)
        else:
            endogenous_data = endogenous_data.values
            exogenous_data = exogenous_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.endogenous_data_x = endogenous_data[border1:border2]
        self.exogenous_data_x = exogenous_data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):

        # exogenous part ,only iTransformer , no label_len
        s_exogenous_begin = index
        s_exogenous_end = s_exogenous_begin + self.seq_len
        r_exogenous_begin = s_exogenous_end
        r_exogenous_end = r_exogenous_begin + self.pred_len
        seq_exogenous_x = self.exogenous_data_x[s_exogenous_begin:s_exogenous_end]

        # endogenous part, itransformer and PatchTST, label_len no use，因为最后还是全连接
        s_endogenous_begin = index
        s_endogenous_end = s_endogenous_begin + self.seq_len
        r_endogenous_begin = s_endogenous_end - self.label_len
        r_endogenous_end = r_endogenous_begin + self.label_len + self.pred_len

        seq_endogenous_x = self.endogenous_data_x[s_endogenous_begin:s_endogenous_end]
        seq_endogenous_y = self.endogenous_data_x[r_endogenous_begin:r_endogenous_end]
        seq_x_mark = self.data_stamp[s_endogenous_begin:s_endogenous_end]
        seq_y_mark = self.data_stamp[r_endogenous_begin:r_endogenous_end]

        return seq_exogenous_x, seq_endogenous_x, seq_endogenous_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.exogenous_data_x) - self.seq_len - self.pred_len + 1

    def inverse_endogenous_transform(self, data):
        return self.scaler_endogenous.inverse_transform(data)

    def inverse_exogenous_transform(self, data):
        return self.scaler_exogenous.inverse_transform(data)

class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train',
                 features='S', data_path='electricity.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.args = args
        # info
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        # init

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.config = configparser.ConfigParser()
        self.config.read(args.config_path)
        self.endogenous_list = ['OT']#self.config.get('Endogenous', 'target').split(',')
        self.exogenous_list = [str(i) for i in range(320)]#self.config.get('Exogenous', 'target').split(',')
        self.patch_len = args.patch_len
        self.scaler_endogenous = StandardScaler()
        self.scaler_exogenous = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)  # 这里面没有考虑pred的那部分，相当于num_train就是单纯的sample的个数
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        # border1s = [0,num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali, len(df_raw)]#从第一个值是num_train可以看到没有考虑pred的部分，这部分落在了vali里面，感觉验证集合训练集还是重合了

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        endogenous_data = df_raw[self.endogenous_list]
        exogenous_data = df_raw[self.exogenous_list]

        if self.scale:
            endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
            exogenous_train_data = exogenous_data[border1s[0]:border2s[0]]
            self.scaler_endogenous.fit(endogenous_train_data.values)
            self.scaler_exogenous.fit(exogenous_train_data.values)
            endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
            exogenous_data = self.scaler_exogenous.transform(exogenous_data.values)
        else:
            endogenous_data = endogenous_data.values
            exogenous_data = exogenous_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.endogenous_data_x = endogenous_data[border1:border2]
        self.exogenous_data_x = exogenous_data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):

        # exogenous part ,only iTransformer , no label_len
        s_exogenous_begin = index
        s_exogenous_end = s_exogenous_begin + self.seq_len
        r_exogenous_begin = s_exogenous_end
        r_exogenous_end = r_exogenous_begin + self.pred_len
        seq_exogenous_x = self.exogenous_data_x[s_exogenous_begin:s_exogenous_end]

        # endogenous part, itransformer and PatchTST, label_len no use，因为最后还是全连接
        s_endogenous_begin = index
        s_endogenous_end = s_endogenous_begin + self.seq_len
        r_endogenous_begin = s_endogenous_end - self.label_len
        r_endogenous_end = r_endogenous_begin + self.label_len + self.pred_len

        seq_endogenous_x = self.endogenous_data_x[s_endogenous_begin:s_endogenous_end]
        seq_endogenous_y = self.endogenous_data_x[r_endogenous_begin:r_endogenous_end]
        seq_x_mark = self.data_stamp[s_endogenous_begin:s_endogenous_end]
        seq_y_mark = self.data_stamp[r_endogenous_begin:r_endogenous_end]

        return seq_exogenous_x, seq_endogenous_x, seq_endogenous_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.exogenous_data_x) - self.seq_len - self.pred_len + 1

    def inverse_endogenous_transform(self, data):
        return self.scaler_endogenous.inverse_transform(data)

    def inverse_exogenous_transform(self, data):
        return self.scaler_exogenous.inverse_transform(data)


