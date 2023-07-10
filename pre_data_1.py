from astropy.io import fits
import numpy as np
import os
from scipy import ndimage
import tensorflow as tf


def per_data1(temp):
    """
    对单个小立方体数据进行预处理：像素值归一化, 尺寸归一化
    :param temp: 小立方体数据
    :return: 处理后的数据，尺寸为30*30*30，值在0-1之间
    验证通过，2019/11/17
    """
    X = []
    # 计算缩放因子
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    scaling_factor = np.array([30 / i for i in temp.shape]).min()
    # 体积归一化
    result = ndimage.zoom(temp, (scaling_factor, scaling_factor, scaling_factor))
    [temp_x, temp_y, temp_z] = result.shape
    pad_x, pad_y, pad_z = (30 - temp_x) // 2, (30 - temp_y) // 2, (30 - temp_z) // 2
    temp1 = np.pad(result, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), 'constant')

    [temp_x, temp_y, temp_z] = temp1.shape
    temp1 = np.pad(temp1, ((30 - temp_x, 0), (30 - temp_y, 0), (30 - temp_z, 0)), 'constant')

    X.append(temp1)
    X = np.array(X)
    return X


def per_fits_data(path):
    """
    做数据预处理
    """
    scale=30
    filelist_ = os.listdir(path)
    X = []
    Y = []
    for i, item in enumerate(filelist_):
        lab = item[0]
        if lab == '1':
            Y.append(1)
        elif lab == '0':
            Y.append(0)
        item_path=os.path.join(path, item)
        temp = fits.getdata(item_path)#读取fits文件
        # print(temp.shape)
        # 做局部归一化
        # temp = (temp - temp.min()) / (temp.max() - temp.min())
        # 计算缩放因子
        scaling_factor = np.array([scale / i for i in temp.shape]).min()
        # 体积归一化
        # temp = ndimage.zoom(temp, (scaling_factor, scaling_factor, scaling_factor))
        if max(temp.shape) >=scale:
        # 体积归一化
            temp = ndimage.zoom(temp, (scaling_factor, scaling_factor, scaling_factor))
        [temp_x, temp_y, temp_z] = temp.shape
        pad_x, pad_y, pad_z = (scale - temp_x) // 2, (scale - temp_y) // 2, (scale - temp_z) // 2
        temp1 = np.pad(temp, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), 'constant')
        [temp_x, temp_y, temp_z] = temp1.shape
        temp1 = np.pad(temp1, ((scale - temp_x, 0), (scale - temp_y, 0), (scale - temp_z, 0)), 'constant')

        bg_data = fits.getdata('data/bg.fits')
        temp1 = temp1 + bg_data
        temp1 = (temp1 - temp1.min()) / (temp1.max() - temp1.min())
        X.append(temp1)

    Y = np.array(Y)
    X = np.array(X)

    return X, Y


if __name__ == '__main__':
    pass