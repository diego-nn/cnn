import numpy as np 
import pandas as pd 
import glob
from PIL import Image
import glob
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, LSTM, TimeDistributed, ZeroPadding2D, BatchNormalization, MaxPooling2D
from keras import Input, Model
from keras.preprocessing.image import ImageDataGenerator

def create_df():
    col_1 = []
    col_2 = []
    col_3 = []
    col_4 = []
    col_5 = []
    col_6 = []
    col_7 = []
    col_8 = []
    for i in sorted(glob.glob('/home/diego/my_project_dir/my_proj_env/flowfrom/images/*.png')): 
        col_1.append(i)
        col_2.append(i)
    col_1.pop()
    col_2.pop(0)
    labels = np.load('/home/diego/Downloads/dataset/poses/01_dl.npy')
    for i in range(len(labels)):
        col_3.append(labels[i][0].astype(float))
        col_4.append(labels[i][1].astype(float))
        col_5.append(labels[i][2].astype(float))
        col_6.append(labels[i][3].astype(float))
        col_7.append(labels[i][4].astype(float))
        col_8.append(labels[i][5].astype(float))
    col_3.pop(0)
    col_4.pop(0)
    col_5.pop(0)
    col_6.pop(0)
    col_7.pop(0)
    col_8.pop(0)

    pd_1 = pd.DataFrame(list(zip(col_1, col_3, col_4, col_5, col_6, col_7, col_8)), columns=['id_1', 'x', 'y', 'z', 'qx', 'qy', 'qz'])
    pd_2 = pd.DataFrame(list(zip(col_2, col_3)), columns=['id_2', 'labels'])
    return pd_1, pd_2

df_1, df_2 = create_df()
in_gen_1 = ImageDataGenerator()
in_gen_2 = ImageDataGenerator()
in_gen_1 = in_gen_1.flow_from_dataframe(df_1, directory='/home/diego/my_project_dir/my_proj_env/flowfrom/images/', x_col="id_1", y_col=['x', 'y', 'z', 'qx', 'qy', 'qz'], target_size=(100,300), batch_size=2, shuffle=False, class_mode='multi_output', color_mode='rgb')
in_gen_2 = in_gen_2.flow_from_dataframe(df_2, directory='/home/diego/my_project_dir/my_proj_env/flowfrom/images/', x_col="id_2", y_col='labels', target_size=(100,300), batch_size=2, shuffle=False, class_mode='raw', color_mode='rgb')

def combine_generator(gen1, gen2):
    while True:
        X1 = in_gen_1.next()
        X2 = in_gen_2.next()
        yield np.stack((X1[0], X2[0]), axis=1), np.stack((X1[1][0], X1[1][1], X1[1][2], X1[1][3], X1[1][4], X1[1][5]), axis=1)

train_generator = combine_generator(in_gen_1, in_gen_2)

tf.test.is_gpu_available(cuda_only=True)
tf.test.gpu_device_name()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.9
sess = tf.compat.v1.Session(config=config)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

cnn = Sequential()
# CONV 1
cnn.add(ZeroPadding2D(padding=3, input_shape=(100, 300, 3)))
cnn.add(Conv2D(64, kernel_size=7, strides=2, padding='valid', activation='relu'))
# CONV 2
#cnn.add(ZeroPadding2D(padding=2))
#cnn.add(Conv2D(128, kernel_size=5, strides=2, padding='valid', activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(MaxPooling2D())
cnn.add(MaxPooling2D())
cnn.add(MaxPooling2D())
# CONV 3
#cnn.add(ZeroPadding2D(padding=2))
#cnn.add(Conv2D(256, kernel_size=5, strides=2, padding='valid', activation='relu'))
# CONV 3_1
#cnn.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
# CONV 4
#cnn.add(ZeroPadding2D(padding=1))
#cnn.add(Conv2D(512, kernel_size=3, strides=2, padding='valid', activation='relu'))
# CONV 4_1
#cnn.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
# CONV 5
#cnn.add(ZeroPadding2D(padding=1))
#cnn.add(Conv2D(512, kernel_size=3, strides=2, padding='valid', activation='relu'))
# CONV 5_1
#cnn.add(Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'))
# CONV 6
#cnn.add(ZeroPadding2D(padding=1))
#cnn.add(Conv2D(1024, kernel_size=3, strides=2, padding='valid'))
# FLATTEN
cnn.add(Flatten())
# LSTM
model = Sequential()
#model.add(TimeDistributed(cnn, input_shape=(2, 376, 1241, 3)))
model.add(TimeDistributed(cnn, input_shape=(2, 100, 300, 3)))
model.add(LSTM(6, return_sequences=False))
model.add(Dense(6))

model.compile(loss='mse', optimizer='adagrad', metrics=['accuracy'])
model.summary()

model.fit_generator(train_generator, steps_per_epoch=1100/8, epochs=10, verbose=1)
