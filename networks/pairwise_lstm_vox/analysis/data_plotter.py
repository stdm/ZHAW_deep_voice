import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

from common.utils.paths import *

Path = '../data/experiments/logs/sess_1494417541/'

file0 = get_experiments("plots", 'sess_1495747514', 'csv_2017-05-25_21-23-42.csv')
# file1 = 'keras_BLSTM64_2Layer_100sp.csv'
# file2 = 'keras_BLSTM32_2Layer_100sp.csv'
# file3 = 'BiLSTM256_l2_sp100.csv'


df = pd.read_csv(file0)
acc = df['acc']
loss = df['loss']
val_acc = df['val_acc']
val_loss = df['val_loss']

# df = pd.read_csv(Path+file1)
# acc1 =  df['acc']
# loss1 = df['loss']
# val_acc1 = df['val_acc']
# val_loss1 = df['val_acc']
#
# df = pd.read_csv(Path+file2)
# acc2 =  df['acc']
# loss2 = df['loss']
# val_acc2 = df['val_acc']
# val_loss2 = df['val_acc']
#
# df = pd.read_csv(Path+file3)
# acc3 =  df['acc']
# loss3 = df['loss']
# val_acc3 = df['val_acc']
# val_loss3 = df['val_loss']



# val_acc = np.array(val.items(), dtype=dtype)
print(val_acc.shape)
v_acc_arr = np.array(val_acc)
# v_acc_arr1 = np.array(val_acc1)
# v_acc_arr2 = np.array(val_acc2)
# v_acc_arr3 = np.array(val_acc3)

x_acc = np.linspace(0, len(val_acc), num=len(val_acc), endpoint=False)
spl_acc = UnivariateSpline(x_acc, v_acc_arr)
spl_acc.set_smoothing_factor(1.8)

# x_acc1 = np.linspace(0, len(val_acc), num=len(val_acc), endpoint=False)
# spl_acc1 = UnivariateSpline(x_acc1,v_acc_arr1)
# spl_acc1.set_smoothing_factor(1.8)
#
# x_acc2 = np.linspace(0, len(val_acc), num=len(val_acc), endpoint=False)
# spl_acc2 = UnivariateSpline(x_acc2,v_acc_arr2)
# spl_acc2.set_smoothing_factor(1.8)
#
# x_acc3 = np.linspace(0, len(val_acc), num=len(val_acc), endpoint=False)
# spl_acc3 = UnivariateSpline(x_acc3,v_acc_arr3)
# spl_acc3.set_smoothing_factor(1.8)

x_train_loss = np.linspace(0, len(loss), num=len(loss), endpoint=False)
spl_train = UnivariateSpline(x_train_loss, loss)
spl_train.set_smoothing_factor(0.2)

x_valid_loss = np.linspace(0, len(val_loss), num=len(val_loss), endpoint=False)
spl_valid = UnivariateSpline(x_valid_loss, val_loss)
spl_valid.set_smoothing_factor(20)

type = "loss"

# sav = "keras_biLSTM_1Layer_128_40sp.png"
if type == "acc":
    # plt.plot( spl_acc3(x_acc3), label='BLSTM 256', color='r')
    plt.plot(spl_acc(x_acc), label='CNN-GRU 128', color='b')
    # plt.plot( spl_acc1(x_acc1), label='BLSTM 64', color='g')
    # plt.plot( spl_acc2(x_acc2), label='BLSTM 32', color='y')
    plt.plot(val_acc, color='b', alpha=0.2)
    # plt.plot(val_acc1, color='g', alpha = 0.2)
    # plt.plot(val_acc2, color='y', alpha = 0.2)
    # plt.plot(val_acc3, color='r', alpha = 0.2)
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    # plt.legend(['BLSTM 256', 'BLSTM 128', 'BLSTM64', 'BLSTM32'] , loc='lower right')
    plt.legend(['CNN-GRU 128'], loc='lower right')
    plt.grid()
    plt.show()

# loss plot
if type == "loss":
    print("loss")
    plt.plot(spl_valid(x_valid_loss), label='Training loss', color='g')
    plt.plot(spl_train(x_train_loss), label='Validation loss', color='b')
    plt.plot(val_loss, color='g', alpha=0.2)
    plt.plot(loss, color='b', alpha=0.2)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.grid()
    plt.show()
