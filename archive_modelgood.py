import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from keras import models


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9)

def get_model(shape, loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['accuracy']):
    
    model = models.Sequential()

    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(shape,1), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(256, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))

    
    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))




    # model = models.Sequential()

    # model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(shape,1), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(MaxPooling1D(pool_size=2))

    # model.add(Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    
    # model.add(Conv1D(256, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))

    # model.add(Conv1D(512, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    
    # model.add(Flatten())

    # model.add(Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(Dropout(0.5))
    # model.add(Dense(8, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
   
    return model