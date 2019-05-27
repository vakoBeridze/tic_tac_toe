import os

import keras
import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential

IMAGES_DIR = "gen_images"
SPLIT_RATE = .8

batch_size = 128
# input image dimensions
img_rows, img_cols = 30, 30


def load_images(images_path=IMAGES_DIR):
    images = []
    labels = []
    for root_dir, dir_names, file_names in os.walk(images_path, topdown=False):
        for filename in file_names:
            label = root_dir.split("/")[-1]
            file_path = os.path.join(root_dir, filename)
            image = Image.open(file_path).convert('L')
            img_data = np.asarray(image)
            img_data = 1 - img_data / 255.0
            images.append(img_data)
            labels.append(1 if label == 'X' else 0)
    return images, labels


def prepare_data(x, y, split_rate=SPLIT_RATE):
    split_rate = int(len(x) * split_rate)
    indices = np.random.permutation(len(x))
    training_idx, test_idx = indices[:split_rate], indices[split_rate:]
    x = np.asarray(x)
    y = np.asarray(y)
    x_training, x_test = x[training_idx, :], x[test_idx, :]
    y_training, y_test = y[training_idx], y[test_idx]
    return (x_training, y_training), (x_test, y_test)


def create_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(img_rows, img_cols, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation="softmax"))
    return model


if __name__ == '__main__':
    x, y = load_images()
    (x_training, y_training), (x_test, y_test) = prepare_data(x, y)

    x_training = x_training.reshape(x_training.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # convert class vectors to binary class matrices
    y_training = keras.utils.to_categorical(y_training, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    print('loaded data x_training={} y_training={}'.format(x_training.shape, y_training.shape))
    print('loaded data x_test={} y_test={}'.format(x_test.shape, y_test.shape))

    model = create_model()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_training, y_training,
              batch_size=batch_size,
              epochs=50,
              verbose=1,
              validation_split=.2)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights("model/weights.h5")
