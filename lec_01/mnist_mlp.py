# Section 1
# Imports
from __future__ import print_function
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
np.random.seed(0)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# Section 2
# Settings
batch_size = 128
num_classes = 10
epochs = 20
loss = 'categorical_crossentropy'
optimizer = RMSprop()
metrics = ['accuracy']


# Section 3
# Train and Test Data
## the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

## change total number
x_train = x_train[range(10000)]
y_train = y_train[range(10000)]
x_test = x_test[range(2000)]
y_test = y_test[range(2000)]

## pre-process
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

## convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## show number of train and test sets
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Section 4
# Check train and test data
## label check
print('Train and Test data 25 examples')
print('Train label :', y_train[range(9)].argmax(1))
print('Test label :', y_test[range(9)].argmax(1))

## plot images
fig = plt.figure(figsize=(3, 3))
for i in range(9):
    fig_sub = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
    fig_sub.imshow(x_train[i].reshape((28,28)), cmap='gray')

fig = plt.figure(figsize=(3, 3))
for i in range(9):
    fig_sub = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
    fig_sub.imshow(x_test[i].reshape((28,28)), cmap='gray')


# Section 5
# Model definition
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

## show model structure
model.summary()


# Section 6
# Train model
## set training options
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


# Section 7
# Check accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1]*100, '％ correct.')


# Section 8
# Plot graph
x_epochs = range(20)
plt.plot(x_epochs, history.history['acc'], label='train')
plt.plot(x_epochs, history.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
