from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report

def plot_confusion_matrix(cm, classes,normalize=True,title=None,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    classes = sorted(classes)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax = ax)
    ax.set(title=title,ylabel='True label',xlabel='Predicted label')

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/hyung.lee/cs229fall2019/speech_commands_threelabels/"
batch_size = 128
num_classes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
epochs = 200
input_shape = (20, 11, 1)

x_train, x_validate, x_test, y_train, y_validate, y_test = get_train_test_valid(path=DATA_PATH)
#model = keras.models.load_model("models/" + "cnnmodel2.h5")
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *input_shape)
y_train_hot = to_categorical(y_train)
y_validate_hot = to_categorical(y_validate)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validate, y_validate_hot))
model.save(DATA_PATH + "cnnmodel2tenwords.h5")
predictions = model.predict_classes(x_test.reshape(x_test.shape[0], *input_shape), batch_size=batch_size, verbose=1)
print("test accuracy:")
print(np.sum(predictions == y_test)/predictions.size)
cnf_matrix = confusion_matrix(y_test, predictions)
class_names = ['backward','cat','five','go','happy','learn','marvin','right','sheila','zero']#['one','two','three','four','five','six','seven','eight','nine','zero']
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
plot_name = 'ConfusionMatrix_Normalized_' + 'Test_' + 'Set.pdf'
plt.savefig(os.path.join('.', plot_name))
print(classification_report(y_test, predictions, target_names=class_names))
