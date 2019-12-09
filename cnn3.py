from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.utils import to_categorical
import numpy as np

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/hyung.lee/cs229fall2019/speech_commands_tenlabels/"
batch_size = 32
num_classes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
epochs = 30
input_shape = (20, 11, 1)

x_train, x_validate, x_test, y_train, y_validate, y_test = get_train_test_valid(path=DATA_PATH)
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *input_shape)
y_train_hot = to_categorical(y_train)
y_validate_hot = to_categorical(y_validate)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train_hot,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_validate, y_validate_hot),
              shuffle=True)

model.save(DATA_PATH + "cnnmodel3.h5")
predictions = model.predict_classes(x_test.reshape(x_test.shape[0], *input_shape), batch_size=batch_size, verbose=1)
print("test accuracy:")
print(np.sum(predictions == y_test)/predictions.size)
