import scipy.constants as constants
import pandas as pds
import datetime
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
import os
import random
import tensorflow as tf
import keras as K

class _Data(object):
    def __init__(self):
        self.windows = 64
        self.evtList = self.read_csv(os.getcwd() + '/data/listOfICMEs.csv', index_col=None)

    def loadParquet(self, filename):
        return pq.read_table(filename).to_pandas()

    def ZscoreNormalization(self, x, mean_, std_):
        """Z-score normaliaztion"""
        x = (x - mean_) / std_
        return x

    # 数据预处理
    def data_pre(self):
        data_ori = self.loadParquet(os.getcwd() + '/data/datasetWithSpectro.parquet')
        data_ori.fillna(0)
        # print(data_ori.isnull().values.any())

        self.computeBeta(data_ori)
        self.computePdyn(data_ori)
        self.computeRmsBob(data_ori)
    
        startTime = datetime.datetime(1997, 10, 1)
        data_ori.index = data_ori.index.tz_localize(None)  # 去除原有时区信息
        endTime = pds.Timestamp(datetime.datetime(2016, 1, 1)).tz_localize(None)  # 去除原有时区信息

        data = data_ori[data_ori.index < endTime]
        data = data[data.index > startTime]
        # data = data.resample('10T').mean().dropna()

        # np.save(os.getcwd() + '/data/X.npy', data)
        return data

    def data_div(self):
        data = self.data_pre()

        X_train = data[(data.index < datetime.datetime(2010, 1, 1)) & (data.index >= datetime.datetime(1998, 1, 1))]
        X_val = data[data.index < datetime.datetime(1998, 1, 1)]
        X_test = data[data.index < datetime.datetime(2010, 1, 1)]

        scale1 = StandardScaler()
        scale1.fit(X_train)
        # scale1.fit(X_test)

        X_train = scale1.transform(X_train)
        X_val = scale1.transform(X_val)
        X_test = scale1.transform(X_test)

        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)

        np.save(os.getcwd() + '/data/X_train_origin_1.npy', X_train)
        np.save(os.getcwd() + '/data/X_val_origin_1.npy', X_val)
        np.save(os.getcwd() + '/data/X_test_origin_1.npy', X_test)

        y = pds.read_csv(os.getcwd() + '/data/Event_0_1_origin.csv', index_col=0)

        # 第一划分
        Y_train = y[(data.index < datetime.datetime(2010, 1, 1)) & (data.index >= datetime.datetime(1998, 1, 1))]
        Y_val = y[data.index < datetime.datetime(1998, 1, 1)]
        Y_test = y[data.index > datetime.datetime(2010, 1, 1)]
        Y_train = np.array(Y_train)
        Y_val = np.array(Y_val)
        Y_test = np.array(Y_test)

        np.save(os.getcwd() + '/data/Y_train_origin_1.npy', Y_train)
        np.save(os.getcwd() + '/data/Y_val_origin_1.npy', Y_val)
        np.save(os.getcwd() + '/data/Y_test_origin_1.npy', Y_test)

        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def eventto0_1(self):
        event = pds.read_csv(os.getcwd() + '/data/listOfICMEs.csv', index_col=None)
        event['begin'] = pds.to_datetime(event['begin'], format="mixed")
        event['end'] = pds.to_datetime(event['end'], format="mixed")
        event = np.array(event)
        # event = event[:5]
        y = pds.DataFrame(index=self.data_pre().index)
        y_copy = y.copy()
        y_copy[1] = 0
        m = 0
        num_1 = 0
        num_0 = 0
        for j in range(event.shape[0]):
            x = event[j]
            print(m, ':', x[0], '---->', x[1])
            for i in y[y.index <= x[1]].index:
                # print(i)
                if i > x[0]:
                    y_copy[y.index == i] = 1
                    num_1 += 1
                else:
                    num_0 += 1
                    pass
            m = m + 1
        print(y_copy[y.index < x[1]])
        print('num_1:', num_1)
        print('num_0:', num_0)
        print('Ratio:', num_0/num_1)
        y_copy.to_csv(os.getcwd() + '/data/Event_0_1_origin.csv', sep=',')
        # np.save(os.getcwd() + '/data/Y.npy', y_copy)


    def Data_format_changes(self, data):
        data_list = []
        for i in range(data.shape[0]):
            data_list.append(data[i, :].reshape(1, data[i, :].shape[0]))
        data = np.array(data_list)
        return data

    def read_csv(self, filename, index_col=0, header=0, dateFormat="mixed", sep=','):
        '''
        Consider a  list of events as csv file ( with at least begin and end)
        and return a list of events
        index_col and header allow the correct reading of the current fp lists
        '''
        df = pds.read_csv(filename, index_col=index_col, header=header, sep=sep)
        df['begin'] = pds.to_datetime(df['begin'], format=dateFormat)
        df['end'] = pds.to_datetime(df['end'], format=dateFormat)
        evtList = [Event(df['begin'][i], df['end'][i]) for i in range(0, len(df))]

        print(evtList)
        return evtList

    def computeBeta(self, data):
        '''
        compute the evolution of the Beta for data
        data is a Pandas dataframe
        The function assume data already has ['Np','B','Vth'] features
        '''
        try:
            data['Beta'] = 1e6 * data['Vth'] * data['Vth'] * constants.m_p * data['Np'] * 1e6 * constants.mu_0 / \
                           (1e-18 * data['B'] * data['B'])
        except KeyError:
            print('Error computing Beta,B,Vth or Np'
                  'might not be loaded in dataframe')
        return data

    def computePm(self, data):
        '''
        compute the evolution of the Magnetic pressure for data
        data is a Pandas dataframe
        The function assume data already has 'B' features
        '''
        try:
            data['Pm'] = 1e-18 * data['B'] * data['B'] / (2 * constants.mu_0)
        except KeyError:
            print('Error computing Beta,B,Vth or Np'
                  ' might not be loaded in dataframe')
        return data

    def computeRmsBob(self, data):
        '''
        compute the evolution of the rmsbob instantaneous for data
        data is a Pandas dataframe
        The function assume data already has ['B_rms] features
        '''
        try:
            data['RmsBob'] = np.sqrt(data['Bx_rms'] ** 2 + data['By_rms'] ** 2 + data['Bz_rms'] ** 2) / data['B']
        except KeyError:
            print('Error computing rmsbob,B or rms of components'
                  ' might not be loaded in dataframe')
        return data

    def computePdyn(self, data):
        '''
        compute the evolution of the Beta for data
        data is a Pandas dataframe
        the function assume data already has ['Np','V'] features
        '''
        try:
            data['Pdyn'] = 1e12 * constants.m_p * data['Np'] * data['V'] ** 2
        except KeyError:
            print('Error computing Pdyn, V or Np might not be loaded '
                  'in dataframe')

class Event:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.duration = self.end-self.begin


class generator_conv1d(object):
    def __init__(self, mode='train', batch_size=16, windows=128, ratio=0.75):
        self.mode = mode
        self.batch_size = batch_size  # 第一个维度
        self.windows = windows  # 第二个维度
        self.ratio = ratio

        self.X_train = np.load(os.getcwd() + '/data/X_train_origin_1.npy')
        self.X_val = np.load(os.getcwd() + '/data/X_val_origin_1.npy')
        self.X_test = np.load(os.getcwd() + '/data/X_test_origin_1.npy')
        self.Y_train = np.load(os.getcwd() + '/data/Y_train_origin_1.npy')
        self.Y_val = np.load(os.getcwd() + '/data/Y_val_origin_1.npy')
        self.Y_test = np.load(os.getcwd() + '/data/Y_test_origin_1.npy')

        self.index_train_1, self.index_train_0 = self.get_index(self.Y_train)
        self.index_val_1, self.index_val_0 = self.get_index(self.Y_val)
        self.index_val = self.index_val_0 + self.index_val_1


    def __iter__(self):
        if self.mode == 'train':
            input_train, output_train = [], []
            while 1:
                x, y = self.get_train_data()
                input_train.append(x)
                output_train.append(y)

                if len(output_train) >= self.batch_size:
                    train_y = np.array(output_train)
                    train_x = np.array(input_train)

                    yield (train_x, train_y)
                    input_train, output_train = [], []
        else:
            input_val, output_val = [], []
            while 1:
                x, y = self.get_val_data()
                input_val.append(x)
                output_val.append(y)

                if len(output_val) >= self.batch_size:
                    val_y = np.array(output_val)
                    val_x = np.array(input_val)

                    yield (val_x, val_y)
                    input_val, output_val = [], []

    def get_train_data(self):
        sampling = random.random()
        sampling_augment = random.random()

        if sampling < self.ratio:
            index = random.sample(self.index_train_1, 1)[0]

            x = self.X_train[index - int(self.windows / 2):index + int(self.windows / 2), :]
            y = self.Y_train[index - int(self.windows / 2):index + int(self.windows / 2), :]

            if sampling_augment < 1:
                x = x
                y = y
            else:
                # 数据扩充： 翻转
                x = np.flip(x, axis=0)
                y = np.flip(y, axis=0)

                # # 数据扩充： 加噪声
                # muti_noise = np.random.normal(1, 0.001, (x.shape[0], 1))
                # x *= muti_noise

                # add_noise = np.random.normal(0, 0.001, (x.shape[0], 1))
                # x += add_noise
        else:
            index = random.sample(self.index_train_0, 1)[0]

            x = self.X_train[index - int(self.windows/2):index + int(self.windows/2), :]
            y = self.Y_train[index - int(self.windows/2):index + int(self.windows/2), :]

        return x, y

    def get_val_data(self):
        index = random.sample(self.index_val, 1)[0]
        x = self.X_val[index - int(self.windows/2):index + int(self.windows/2), :]
        y = self.Y_val[index - int(self.windows/2):index + int(self.windows/2), :]
        return x, y

    def get_index(self, file):
        index_1, index_0 = [], []

        for i in range(int(self.windows/2), file.shape[0]-int(self.windows/2)):
            if file[i, -1] == 1:
                index_1.append(i)
            else:
                index_0.append(i)

        return index_1, index_0

if __name__ == '__main__':
    data = _Data()
    # data.eventto0_1()
    # data.data_pre()
    data.data_div()