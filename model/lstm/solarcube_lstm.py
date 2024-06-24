# -*- coding: utf-8 -*-
import os


import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras import layers, models

data = np.load('solarsat_point_long_train.npz', allow_pickle=True)
x_train = data['arr_0']
y_train = data['arr_1']
df_train = data['arr_2']
data = np.load('solarsat_point_long_test.npz', allow_pickle=True)
x_test = data['arr_0']
y_test = data['arr_1']
df_test = data['arr_2']

x_test=np.transpose(x_test[:,0,:,:], (0,2,1))
y_test=np.transpose(y_test[:,0,:,:], (0,2,1))
x_train=np.transpose(x_train[:,0,:,:], (0,2,1))
y_train=np.transpose(y_train[:,0,:,:], (0,2,1))

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)


# Callbacks
bn = 'best_lstm_attention_model_long'
checkpoint = ModelCheckpoint(bn+'.h5', monitor='val_loss', save_best_only=True, mode='min')
csv_logger = CSVLogger('training_log.csv', append=True)

callbacks_list = [checkpoint, csv_logger]

def build_lstm_model():
    # Input layer: Expect input shape (96, 5), where 8 is the number of time steps and 5 is the number of features per time step.
    inputs = keras.Input(shape=(96, 5))

    # LSTM layer: Let's assume we use 50 units and want to keep the sequence
    x = layers.LSTM(128, return_sequences=False)(inputs)

    # Repeat the context vector to match the desired output time steps (12)
    x = layers.RepeatVector(12)(x)

    # LSTM layer: Return sequences for the repeated context vector
    x = layers.LSTM(128, return_sequences=True)(x)

    # TimeDistributed Dense layer to project the output to the desired feature dimension (1)
    outputs = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(1, activation='relu'))(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=optimizer, loss='mse')  # Compile with an optimizer and loss function

    return model




def build_lstm_attention_model():
    # Input layer: Expect input shape (8, 5), where 8 is the number of time steps and 5 is the number of features per time step.
    inputs = tf.keras.Input(shape=(96, 5))

    # Encoder
    encoder_embedded = layers.Dense(256, activation='relu')(inputs)
    encoder_outputs, state_h, state_c = layers.LSTM(256, return_sequences=True, return_state=True)(encoder_embedded)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_embedded = layers.Dense(256, activation='relu')(inputs)
    decoder_outputs, _, _ = layers.LSTM(256, return_sequences=True, return_state=True)(decoder_embedded, initial_state=encoder_states)

    # Attention
    context_vector, attention_weights = layers.AdditiveAttention()([decoder_outputs, encoder_outputs], return_attention_scores=True)
    decoder_combined_context = tf.concat([context_vector, decoder_outputs], axis=-1)

    # Output
    outputs = layers.Dense(1)(decoder_combined_context)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')  # Compile with an optimizer and loss function

    return model


model = build_lstm_attention_model()
model.summary()

# y_train_norm = (y_train-193.44)/279.49
# y_val_norm = (y_val-193.44)/279.49
model.fit(x_train, y_train, epochs=300, batch_size=16, validation_data=(x_val, y_val), callbacks=callbacks_list)

# Load the best model
best_model = keras.models.load_model(bn+'.h5')

# Evaluate the model on the test data
test_loss = best_model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')

# Make predictions on the test data
predictions = best_model.predict(x_test)
# print(predictions)

np.save('attention_lstm_long_pred.npy', predictions)

