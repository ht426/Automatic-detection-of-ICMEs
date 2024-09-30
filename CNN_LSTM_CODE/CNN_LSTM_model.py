from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np

class CNN_LSTM_model(object):
    def __init__(self): 
        self.windows = 64

    def cnn_lstm(self):
        # 输入层
        inputs = layers.Input((self.windows, 33))  # 输入形状为 (None, 64, 33)
        # 添加第一个卷积层
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        # 添加第二个卷积层
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        # 添加第三个卷积层
        x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
        # 添加第四个卷积层
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        # 添加第五个卷积层
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)  # 多了这一层
        # 添加 LSTM 层
        x = layers.LSTM(64, return_sequences=True)(x)
        # 应用 TimeDistributed 层来应用 Dense 层到每个时间步的输出
        y = layers.TimeDistributed(layers.Dense(1, activation='linear'))(x)
        # 创建模型
        model = Model(inputs, y)
        return model

if __name__ == '__main__':
    cnn_lstm_model = CNN_LSTM_model()  # 创建模型实例
    model = cnn_lstm_model.cnn_lstm()  # 调用实例方法
    model.summary()  # 输出模型结构
