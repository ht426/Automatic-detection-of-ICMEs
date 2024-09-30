import os
import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras as K
from Data_origin import generator_conv1d, _Data, Event
from CNN_LSTM_model import CNN_LSTM_model
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from scipy.ndimage import distance_transform_edt as distance
from tensorflow.keras import backend as ba
import datetime
import time
# from torch.nn import *
import random


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)]
)

def re(y_true, y_pred):
    true_positives = K.backend.sum(K.backend.round(K.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.backend.sum(K.backend.round(K.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.backend.epsilon())
    # print(true_positives)
    # print(possible_positives)
    return recall


def prec(y_true, y_pred):
    true_positives = K.backend.sum(K.backend.round(K.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.backend.sum(K.backend.round(K.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.backend.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = prec(y_true, y_pred)
    recall = re(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.backend.epsilon()))


def focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape needs to be (None, 1)
        y_pred needs to be computed after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)

        # 计算 p_t 并增加一个小常数以避免 log(0)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
        p_t = tf.clip_by_value(p_t, tf.keras.backend.epsilon(), 1.0)  # 防止log(0)

        # 计算焦点损失
        focal_loss = - alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    return binary_focal_loss_fixed
    
class Pipeline(object):
    def __init__(self):
        self.windows = 64
        self.batch_size = 16
        # self.ratio = 0.5
        self.model = 'cnn_lstm'
        self.lr = 0.0001
        self.epoch = 100
        # self.thres = 0.5
        self.evtList = _Data().read_csv(os.getcwd() + '/data/listOfICMEs.csv', index_col=None)

    def fit(self):
        train_data = generator_conv1d(mode='train', batch_size=self.batch_size,
                                      windows=self.windows, ratio=ratio)
        val_data = generator_conv1d(mode='val', batch_size=self.batch_size,
                                    windows=self.windows)

        callbacks = []
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                           epsilon=0.001, cooldown=1, verbose=1))
        callbacks.append(ModelCheckpoint('/ICME/model_origin_2/' + self.model + '/' + str(ratio) + '_' + str(self.windows)
                                         + '/model_{epoch:04d}_{val_f1:04f}_{val_re:04f}_{val_prec:04f}_{val_loss:.04f}_{val_accuracy:.04f}_1.hdf5',
                                         monitor='val_f1', verbose=1))

        model =CNN_LSTM_model().cnn_lstm()
        model.compile(loss=focal_loss(), optimizer=optimizers.Adam(lr=self.lr),
                      metrics=[f1, re, prec, 'accuracy'])
        model.summary()

        # model = load_model('/ICME/model_origin_2/' + self.model + '/' + str(ratio) + '_' + str(self.windows) +
        #                    '/model_0100_0.467516_0.439017_0.581120_3.1630_0.8926_1.hdf5',
        #                    custom_objects={'f1': f1, 're': re, 'prec': prec})
        history = model.fit_generator(generator=train_data.__iter__(), epochs=self.epoch, steps_per_epoch=10000,
                                      verbose=1, validation_data=val_data.__iter__(), validation_steps=10000,
                                      callbacks=callbacks, initial_epoch=0)

        return model

    def test(self,):
        model = self.fit()

        X_test = np.load(os.getcwd() + '/data/X_test_origin_2.npy')
        Y_test = np.load(os.getcwd() + '/data/Y_test_origin_2.npy')
        n = X_test.shape[0]
        test_para = X_test.copy()
        test_label = Y_test.copy()

        prediction = []
        for i in range(int(test_para.shape[0]/self.windows)):
            x_test = test_para[i * self.windows:(i + 1) * self.windows, :]
            x_test = x_test.reshape((1, self.windows, 33))
            a = model.predict(x_test)[0]
            prediction.append(a)
            # print(i)
        # print(0, ':')
        x_test = test_para[-self.windows:, :]
        x_test = x_test.reshape((1, self.windows, 33))
        a = model.predict(x_test, verbose=1)[0]
        prediction.append(a[-(test_para.shape[0] - int(test_para.shape[0] / self.windows) * self.windows):, :])
        print('Finish!')

        prediction = np.concatenate(prediction, axis=0)
        # for i in range(prediction.shape[0]):
        #     if prediction[i, 0] > 0.5:
        #         prediction[i, 0] = 1
        #     else:
        #         prediction[i, 0] = 0
        prediction = prediction.astype(int)
        # print(prediction)
        prediction = self.correction(prediction, thres=4)

        pre_label = prediction.copy()
        pre_label = pre_label.reshape((pre_label.shape[0], 1)).astype(int)

        np.save('/ICME/result_origin_2/' + self.model + '/' + str(ratio) + '_' + str(self.windows) + '/100.npy', pre_label)

        # f1 = f1_score(test_label, pre_label)
        # recall = recall_score(test_label, pre_label)
        # precision = precision_score(test_label, pre_label)
        #
        # print('f1_score:', f1)
        # print('recall:', recall)
        # print('precision:', precision)
        #
        # results = classification_report(test_label, pre_label)
        # print(results)

    def eval(self):
        # 获取真实的ICME间隔
        test_clouds = [x for x in self.evtList if x.begin.year > 2009]

        # 获取预测的ICME间隔
        prediction = np.load('/ICME/result_origin_1/' + self.model + '/' + str(ratio) + '_' + str(self.windows) + '/100.npy')
        data_index = _Data().data_pre().index
        data_index = data_index[data_index < datetime.datetime(2003, 1, 1)]

        pre_event = self.getevent(prediction, data_index)
        pre_event = self.mergeEvent(pre_event, thres=2)

        TP_ICME, FN_ICME, FP_ICME, detected_ICME, TP_IOG = self.evaluate(pre_event, test_clouds)

        TP = len(TP_ICME)
        FP = len(FP_ICME)
        FN = len(FN_ICME)
        detected = len(detected_ICME)
        R = TP/(TP+FN)
        P = TP/(TP+FP)
        f1 = (2*R*P)/(R+P)
        IOG_mean = np.mean(TP_IOG)
        print('TP:', TP)
        print('FP:', FP)
        print('FN:', FN)
        print('detected:', detected)
        print('R:', R)
        print('P:', P)
        print('f1:', f1)
        print('IOU_mean:', IOG_mean)
        print('----------------------------------------------------')

    def windowed(self, X, window):
        shape = int((X.shape[0] - window) + 1), window, X.shape[1]
        strides = (X.strides[0],) + X.strides
        X_windowed = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        return X_windowed

    def correction(self, prediction, thres):
        n = 0
        y = prediction.copy()
        for i in range(prediction.shape[0]):
            if prediction[i] != prediction[n]:
                if i - n <= thres:
                    for j in range(n, i):
                        y[j] = prediction[i]
                    n = i
                else:
                    n = i
            else:
                pass
        return y

    def getevent(self, prediction, data_index):
        pre_event = []
        m = 0
        for i in range(prediction.shape[0] - 1):
            dic = {}
            if prediction[i + 1, :] != prediction[i, :]:
                if i + 1 - m > 12:
                    if prediction[m, :] == 1:
                        dic['begin'] = data_index[m]
                        dic['duration'] = data_index[i] - data_index[m]
                        dic['end'] = data_index[i]
                        pre_event.append(dic)
                        m = i + 1
                    else:
                        m = i + 1
                else:
                    m = i + 1
            else:
                pass
        return pre_event

    def mergeEvent(self, pre_event, thres):
        i = 0
        while i < len(pre_event) - 1:
            if pre_event[i + 1]['begin'] - pre_event[i]['end'] < datetime.timedelta(hours
                                                                                    =thres):
                pre_event[i]['begin'] = pre_event[i]['begin']
                pre_event[i]['end'] = pre_event[i + 1]['end']
                pre_event[i]['duration'] = pre_event[i + 1]['end'] - pre_event[i]['begin']
                pre_event.pop(i + 1)
            else:
                i += 1
        return pre_event

    def evaluate(self, predicted_list, test_list, durationCreepies=2.5):
        TP = []
        FN = []
        FP = []
        detected = []
        TP_IOG = []
        for event in test_list:
            correspondings = self.find(event, predicted_list)
            if correspondings is None:
                FN.append(event)
            else:
                TP.append(correspondings)
                TP_IOG.append(self.calculation_iog(correspondings, event))
                detected.append(event)

        FP = [x for x in predicted_list if max(self.overlapWithList(test_list, x)) == 0]
        seum = [x for x in FP if x['duration'] < datetime.timedelta(hours=durationCreepies)]
        for event in seum:
            FP.remove(event)
            predicted_list.remove(event)

        return TP, FN, FP, detected, TP_IOG

    def _evaluate(self, predicted_list, test_list, durationCreepies=2.5):
        TP = []
        FN = []
        FP = []
        detected = []
        TP_IOG = []
        for event in test_list:
            correspondings = self._find(event, predicted_list)
            if correspondings is None:
                FN.append(event)
            else:
                TP.append(correspondings)
                TP_IOG.append(self._calculation_iog(correspondings, event))
                detected.append(event)

        FP = [x for x in predicted_list if max(self.overlapWithList(test_list, x)) == 0]
        seum = [x for x in FP if x['duration'] < datetime.timedelta(hours=durationCreepies)]
        for event in seum:
            FP.remove(event)
            predicted_list.remove(event)

        return TP, FN, FP, detected, TP_IOG

    def find(self, ref_event, event_list):
        if self.isInList(ref_event, event_list):
            return (self.choseEventFromList(ref_event, event_list))
        else:
            return None

    def _find(self, ref_event, event_list):
        if self.isInList(ref_event, event_list):
            return (self._choseEventFromList(ref_event, event_list))
        else:
            return None

    def calculation_iog(self, correspondings, event):
        Sum = datetime.timedelta(0)
        for i in range(len(correspondings)):
            Sum += self.overlap(event, correspondings[i])
        return Sum/event.duration

    def _calculation_iog(self, correspondings, event):
        return self.overlap(event, correspondings)/event.duration

    def isInList(self, ref_event, event_list):
        return max(self.overlapWithList(ref_event, event_list, percent=True)) > 0

    def choseEventFromList(self, ref_event, event_list):
        event = []
        for elt in event_list:
            if self._overlap(ref_event, elt) != None:
                event.append(self._overlap(ref_event, elt))
        return event

    def _choseEventFromList(self, ref_event, event_list):
        return event_list[np.argmax(self.overlapWithList(ref_event, event_list, percent=True))]

    def merge(self, event1, event2):
        return Event(event1.begin, event2.end)

    def _merge(self, event1, event2):
        event = {}
        event['begin'] = event1['begin']
        event['end'] = event2['end']
        event['duration'] = event['end'] - event['begin']
        return event

    def overlapWithList(self, ref_event, event_list, percent=False):
        if percent:
            return [self.overlap(ref_event, elt) / ref_event.duration
                                                    for elt in event_list]
        else:
            return [self.overlap(ref, event_list) / ref.duration
                                                    for ref in ref_event]

    def _overlap(self, event1, event2):
        delta1 = min(event1.end, event2['end'])
        delta2 = max(event1.begin, event2['begin'])
        if delta1 - delta2 > datetime.timedelta(0):
            return event2

    def overlap(self, event1, event2):
        delta1 = min(event1.end, event2['end'])
        delta2 = max(event1.begin, event2['begin'])
        return max(delta1 - delta2, datetime.timedelta(0))

if __name__ == '__main__':
    model_lis = ['Atrous_AttUnet', 'Atrous_Unet', 'AttUnet', 'R2AttUnet', 'R2Unet', 'Unet', 'Unet2P', 'Unet3P']
    ratio = 0.7

    pipeline = Pipeline()
    # pipeline.fit()
    pipeline.test()
    pipeline.eval()