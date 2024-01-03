import numpy as np
import tensorflow as tf
import glob
import os
import seaborn as sns
import csv
import yfinance as yf
import argparse
import tqdm
import time
import matplotlib.pyplot as plt
from datetime import datetime

class AttenLayer(tf.keras.layers.Layer):


    def __init__(self, num_state, **kw):
        super(AttenLayer, self).__init__(**kw)
        self.prob_kernel = None
        self.bias = None
        self.kernel = None
        self.num_state = num_state

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[input_shape[-1], self.num_state])
        self.bias = self.add_weight('bias', shape=[self.num_state])
        self.prob_kernel = self.add_weight('prob_kernel', shape=[self.num_state])

    def call(self, input_tensor, **kwargs):
        atten_state = tf.tanh(tf.tensordot(input_tensor, self.kernel, axes=1) + self.bias)
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
        prob = tf.nn.softmax(logits)
        weighted_feature = tf.reduce_sum(tf.multiply(input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature

    # for saving the model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state, })
        return config
def ReshapeLayer(x):
    
    shape = x.shape
    
    # 1 possibility: H,W*channel
    reshape = tf.keras.layers.Reshape((shape[1],shape[2]*shape[3]))(x)
    
    # 2 possibility: W,H*channel
    # transpose = Permute((2,1,3))(x)
    # reshape = Reshape((shape[1],shape[2]*shape[3]))(transpose)
    
    return reshape

# Xây dựng mô hình Bidirectional LSTM + Attention Layer
def build_model_BiLSTM(num_input, num_output):
  x_in = tf.keras.Input(shape=(num_input, 1))
  x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=30, return_sequences=True))(x_in)
  x_tensor = AttenLayer(20)(x_tensor)
  x_tensor = tf.keras.layers.Dropout(0.2)(x_tensor)
  pred = tf.keras.layers.Dense(num_output, activation='leaky_relu')(x_tensor)
  model = tf.keras.Model(inputs=x_in, outputs=pred)
  return model

# Xây dựng mô hình CNN2D + Bidirectional LSTM
def build_model_CNN_BiLSTM(shape_input, num_output):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(shape_input[1], shape_input[-1], 1), padding='same', use_bias=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Lambda(ReshapeLayer))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=30, return_sequences=True)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(AttenLayer(20))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(num_output, activation='linear'))
    return model

# Xây dựng mô hình LSTM
def build_model_LSTM(num_input, num_output):
    model = tf.keras.Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(num_input, 1)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.5))

    # Adding the output layer
    model.add(tf.keras.layers.Dense(units=num_output))

    # Compiling the RNN
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mean_squared_error")

    return model
