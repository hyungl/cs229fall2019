from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/hyung.lee/cs229fall2019/speech_commands_tenlabels/"

batch_size = 100
num_classes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
print(num_classes)
epochs = 200
input_shape = (20, 11, 1)

x_train, x_validate, x_test, y_train, y_validate, y_test = get_train_test_valid(path=DATA_PATH)
#model = keras.models.load_model(DATA_PATH + "cnnmodel.h5")
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *input_shape)
y_train_hot = to_categorical(y_train)
y_validate_hot = to_categorical(y_validate)
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validate, y_validate_hot))
model.save(DATA_PATH + "cnnmodel.h5")
predictions = model.predict_classes(x_test.reshape(x_test.shape[0], *input_shape), batch_size=100, verbose=1)
print("test accuracy:")
print(np.sum(predictions == y_test)/predictions.size)
