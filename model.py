import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
def get_model(shape, loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule), metrics=['accuracy']):


    model = Sequential()
    model.add(Conv1D(256, 8, padding='same',input_shape=(shape,1)))  
    model.add(Activation('relu'))

    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model