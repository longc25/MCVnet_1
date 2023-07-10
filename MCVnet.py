# -*- coding: utf-8 -*-
# @Time : 2022/8/9 9:55
# @Author : longchen
# @File : resnet.py
# @Project : sort

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model  # 这个用的是tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from random import shuffle
import pre_data_1 as pre_data
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
# from utils import *
import datetime

np.random.seed(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# 定义Basic Block
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # 第一小块
        self.conv1 = layers.Conv3D(filter_num, (3, 3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 第二小块
        self.conv2 = layers.Conv3D(filter_num, (3, 3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv3D(filter_num, (1, 1, 1), strides=stride, padding='same'))
            # self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = layers.add([out, identity])
        out = tf.nn.relu(out)
        return out


'''定义ResNet'''


# 定义ResNet
class ResNet(keras.Model):

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    def __init__(self, layer_dims, num_classes=2):  # mnist有10类,此时2类
        super(ResNet, self).__init__()
        self.stem = Sequential([layers.Input(shape=(30, 30, 30, 1)),
                                layers.Conv3D(16, (3,3, 3), strides=(2,2, 2), padding='same'),  # 15，15，15
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1, 1, 1))
                                ])
        self.layer1 = self.build_resblock(32, layer_dims[0], stride=2)  # 8，8，8

        self.layer2=Sequential([layers.Dropout(0.5),layers.Conv3D(64, (3,3,3), strides=(2,2,2),padding='same')])#4，4，4
        # self.layer2 = self.build_resblock(64, layer_dims[1], stride=2)
        self.layer3 = Sequential([
            # layers.GlobalAveragePooling3D(),
            layers.Flatten(),
            # layers.Dense(2048),  # 原来为1024
            # # layers.Dropout(0.5),
            # layers.Dense(512),
            # layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')])

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)

        x = self.layer1(x, training=training)

        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        # x = self.avgpool(x)

        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x


# def ResNet18():
#     return ResNet([1,1])

def train(train_data,model, log_path, checkpoint_filepath='/tmpeckpoint', batch_size=300, EPOCHS=100):
    # time_callback = TimeHistory()
    log_dir = os.path.join(log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=80, mode='max',
    #                           restore_best_weights=True)

    # X, Y = pre_data.per_fits_data('/home/data/longc/data/data_set_20191116/train')
    X, Y = pre_data.per_fits_data(train_data)
    Y = keras.utils.to_categorical(Y, num_classes=2)

    index_ = [i for i in range(1, X.shape[0], 4)]
    X_test = X[index_, ...]
    Y_test = Y[index_, ...]
    X = np.delete(X, index_, axis=0)
    Y = np.delete(Y, index_, axis=0)

    X = tf.expand_dims(X, -1)
    X = tf.cast(X, dtype=tf.float32)
    X_test = tf.expand_dims(X_test, -1)
    X_test = tf.cast(X_test, dtype=tf.float32)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001,
                                                                   decay_rate=0.96,
                                                                   decay_steps=10000)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    # model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])#categorical_crossentropy
    'binary cross entropy'
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])  # categorical_crossentropy
    # hist = model.fit(X, Y, batch_size=200, epochs=60, verbose=2, validation_split=0.3, shuffle=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.

    hist = model.fit(X, Y, batch_size=batch_size, epochs=EPOCHS, verbose=2, validation_data=(X_test, Y_test),
                     shuffle=True, callbacks=[tf_callback,model_checkpoint_callback])

    """对测试集进行预测"""
    Y_test_pred = np.argmax(model.predict(X_test), axis=1)
    Y_test_pred_hot = keras.utils.to_categorical(Y_test_pred, num_classes=2)
    print('Test accuracy: {:.3f}'.format(accuracy_score(Y_test, Y_test_pred_hot)))

    print('the train data set has done!')
    return hist, model, X, Y


# 独热编码
def _OneHot_decode(encoded_data):
    # encoded_data = _OneHot_encode()
    decoded_data = []
    for i in range(encoded_data.shape[0]):
        decoded = np.argmax(encoded_data[i], axis=0)
        decoded_data.append(decoded)
    decoded_data = np.array(decoded_data)
    # print(decoded_data.shape)
    return decoded_data


# 画损失函数图
def training_vis(hist, savepath):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']  # new version => hist.history['accuracy']
    val_acc = hist.history['val_accuracy']  # => hist.history['val_accuracy']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    # 这步是关键
    # 坐标轴的刻度设置向内(in)或向外(out)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax1 = fig.add_subplot(121)
    # 设置主副坐标轴
    ax1.tick_params(top=True, right=True)

    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    # ax1.set_title('Loss on Training and Validation Data')

    # ax1.set_title('Loss on Training  Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.tick_params(top=True, right=True)
    ax2.plot(acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    # ax2.set_title('Accuracy  on Training and Validation Data')
    # 坐标轴的刻度设置向内(in)或向外(out)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # ax2.set_title('Accuracy  on Training Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(savepath, dpi=600, format='png')


# def main():
#     save_dir = '/home/data/longc/pycharm/voxnet/realdata_result/date_1024'
#     os.makedirs(save_dir, exist_ok=True)
#     log_savepath=os.path.join(save_dir, 'mylogs')
#     model = ResNet([1])
#
#     hist, model, X, Y = train(model, log_savepath, checkpoint_filepath='/tmpeckpoint', batch_size=300)
#
#     fig_savepath = os.path.join(save_dir, 'training_loss.eps')
#     training_vis(hist, fig_savepath)
#     # 权重及history的保存：
#     model_savepath = os.path.join(save_dir, 'model')
#     model.save(model_savepath)
#     training_savepath = os.path.join(save_dir, 'simulate_log.csv')
#     pd.DataFrame(hist.history).to_csv(training_savepath, index=False)  # 保存日志
#     print(model.summary())
#     model_savepath1 = os.path.join(save_dir, 'model.h5')
#     model.save_weights(model_savepath1)

def test():
    # 预测
    #m16
    # model =load_model(r'/home/data/longc/pycharm/voxnet/realdata_result/date_04011/model')
    # path=r'/home/data/longc/pycharm/voxnet/data_set_20191116/test'
    #R2
    model = load_model(r'/home/data/longc/pycharm/voxnet/R2_R16_3/training_result/date_0413/model/')
    #R2_test
    path = r'/home/data/longc/pycharm/voxnet/R2_R16_data1/data/test_R2'
    #R16
    # path=r'/home/data/longc/pycharm/voxnet/R2_16_data/R16/uncertain_sample/'
    # path = r'/home/data/longc/pycharm/voxnet/R2_16_data/test_data/'
    X_predicate, Y_predicate = pre_data.per_fits_data(path)
    X_predicate = tf.convert_to_tensor(X_predicate)
    X_predicate = tf.expand_dims(X_predicate, -1)
    Y_test_pred = np.argmax(model.predict(X_predicate), axis=1)
    Y_test_pred_hot = keras.utils.to_categorical(Y_test_pred, num_classes=2)
    Y_predicate_hot = keras.utils.to_categorical(Y_predicate, num_classes=2)
    #
    print('Test accuracy: {:.3f}'.format(accuracy_score(Y_predicate_hot, Y_test_pred_hot)))
    #
    """# Show confusion matrix and average per-class accuracy"""
    conf = confusion_matrix(Y_predicate, Y_test_pred)
    # plt.figure(figsize=(15, 5))
    # # sns.heatmap(confusion_matrix(Y_predicate, Y_test_pred), annot=True, cmap='Blues')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title("Resnet3D")
    # plt.show()
    avg_per_class_acc = np.mean(np.diagonal(conf) / np.sum(conf, axis=1))
    print('Confusion matrix:\n{}'.format(conf))
    print('Average per-class accuracy: {:.3f}'.format(avg_per_class_acc))



# def get_lendata():
#     X, Y = pre_data.per_fits_data('/home/data/longc/data/data_set_20191116/train')
#
#     Y = keras.utils.to_categorical(Y, num_classes=2)
#
#     index_ = [i for i in range(1, X.shape[0], 4)]
#     X_test = X[index_, ...]
#     Y_test = Y[index_, ...]
#     X = np.delete(X, index_, axis=0)
#     Y = np.delete(Y, index_, axis=0)
#
#     X = tf.expand_dims(X, -1)
#
#     X_test = tf.expand_dims(X_test, -1)

def second_training(model, save_dir, batch_size=300, EPOCHS=100):
    model = load_model(model)

    os.makedirs(save_dir, exist_ok=True)
    log_savepath = os.path.join(save_dir, 'mylogs')
    checkpoint_filepath = os.path.join(save_dir, 'ckp')
    os.makedirs(checkpoint_filepath, exist_ok=True)

    tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log_savepath)

    X, Y = pre_data.per_fits_data('/home/data/longc/data/data_set_20191116/train')
    Y = keras.utils.to_categorical(Y, num_classes=2)

    index_ = [i for i in range(1, X.shape[0], 4)]
    X_test = X[index_, ...]
    Y_test = Y[index_, ...]
    X = np.delete(X, index_, axis=0)
    Y = np.delete(Y, index_, axis=0)

    X = tf.expand_dims(X, -1)

    X_test = tf.expand_dims(X_test, -1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)
    hist = model.fit(X, Y, batch_size=batch_size, epochs=EPOCHS, verbose=2, validation_data=(X_test, Y_test),
                     shuffle=True,
                     callbacks=[tf_callback, model_checkpoint_callback])

    fig_savepath = os.path.join(save_dir, 'training_loss.png')
    training_vis(hist, fig_savepath)
    # 权重及history的保存：
    model_savepath = os.path.join(save_dir, 'model')
    model.save(model_savepath)
    training_savepath = os.path.join(save_dir, 'simulate_log.csv')
    pd.DataFrame(hist.history).to_csv(training_savepath, index=False)  # 保存日志
    print(model.summary())
    model_savepath1 = os.path.join(save_dir, 'model.h5')
    model.save_weights(model_savepath1)
    """对测试集进行预测"""
    Y_test_pred = np.argmax(model.predict(X_test), axis=1)
    Y_test_pred_hot = keras.utils.to_categorical(Y_test_pred, num_classes=2)
    print('Test accuracy: {:.3f}'.format(accuracy_score(Y_test, Y_test_pred_hot)))

    print('the train data set has done!')


if __name__ == '__main__':
    #训练
    #m16
    # train_data = r'/home/data/longc/pycharm/voxnet/data_set_20191116/train/'
    # save_dir = r'/home/data/longc/pycharm/voxnet/realdata_result/date_04011'
    #R2
    train_data=r'/home/data/longc/pycharm/voxnet/R2_R16_3/R2/train_data_1'
    save_dir = r'/home/data/longc/pycharm/voxnet/R2_R16_3/training_result/date_0605_bg'
    os.makedirs(save_dir, exist_ok=True)
    log_savepath = os.path.join(save_dir, 'mylogs')
    os.makedirs(log_savepath, exist_ok=True)
    checkpoint_filepath = os.path.join(save_dir, 'ckp')
    os.makedirs(checkpoint_filepath, exist_ok=True)

    model = ResNet([1])
    #m16
    # hist, model, X, Y = train(train_data, model, log_savepath, checkpoint_filepath=checkpoint_filepath, batch_size=400,EPOCHS=100)
    # #R2
    hist, model, X, Y = train(train_data,model, log_savepath, checkpoint_filepath=checkpoint_filepath, batch_size=400, EPOCHS=300)
    #
    fig_savepath = os.path.join(save_dir, 'training_loss.png')
    training_vis(hist, fig_savepath)
    # 权重及history的保存：
    model_savepath = os.path.join(save_dir, 'model')
    model.save(model_savepath)
    training_savepath = os.path.join(save_dir, 'simulate_log.csv')
    pd.DataFrame(hist.history).to_csv(training_savepath, index=False)  # 保存日志
    print(model.summary())
    model_savepath1 = os.path.join(save_dir, 'model.h5')
    model.save_weights(model_savepath1)

    #预测
    # test()

    # 二次训练
    # model_path=r'/home/data/longc/pycharm/voxnet/realdata_result/date_1024_thirdtraining/model/'
    # save_dir = '/home/data/longc/pycharm/voxnet/realdata_result/date_1024_forthtraining'
    # second_training(model_path,save_dir, batch_size=300, EPOCHS=200)

#仿真
    # save_dir = '/home/data/longc/pycharm/voxnet/simulate_result/resnet/date_1207'
    # os.makedirs(save_dir, exist_ok=True)
    # log_savepath = os.path.join(save_dir, 'mylogs')
    # os.makedirs(log_savepath, exist_ok=True)
    # checkpoint_filepath = os.path.join(save_dir, 'ckp')
    # os.makedirs(checkpoint_filepath, exist_ok=True)
    #
    # train_data =r'/home/data/longc/pycharm/voxnet/simulate_data/train'
    # model = ResNet([1])
    #
    # hist, model, X, Y = train(train_data,model, log_savepath, checkpoint_filepath=checkpoint_filepath, batch_size=400, EPOCHS=200)
    #
    # fig_savepath = os.path.join(save_dir, 'training_loss.png')
    # training_vis(hist, fig_savepath)
    # # 权重及history的保存：
    # model_savepath = os.path.join(save_dir, 'model')
    # model.save(model_savepath)
    # training_savepath = os.path.join(save_dir, 'simulate_log.csv')
    # pd.DataFrame(hist.history).to_csv(training_savepath, index=False)  # 保存日志
    # print(model.summary())
    # model_savepath1 = os.path.join(save_dir, 'model.h5')
    # model.save_weights(model_savepath1)