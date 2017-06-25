import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import np_utils
from keras import optimizers
from keras import callbacks
from keras import metrics

from pathlib import Path


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('log_name', type=click.Path())
def main(input_filepath, output_filepath, log_name):

    # create output folder if not exists
    Path(output_filepath).mkdir(parents=True, exist_ok=True)

    bottleneck_features_path = input_filepath
    top_model_weights_path = output_filepath + '/bottleneck_weights_model.h5'
    top_model_path = output_filepath + '/bottleneck_model.h5'
    nb_train_samples = 1852
    nb_validation_samples = 468
    nb_classes = 6
    epochs = 500
    batch_size = 16

    def train_top_model():
        train_data = np.load(open(bottleneck_features_path + 'bottleneck_features_train.npy', 'rb'))
        train_data_len = len(train_data)
        train_labels_not_encoded = np.array(
            [0] * (nb_train_samples // nb_classes) +
            [1] * (nb_train_samples // nb_classes) +
            [2] * (nb_train_samples // nb_classes) +
            [3] * (nb_train_samples // nb_classes) +
            [4] * (nb_train_samples // nb_classes) +
            [5] * (nb_train_samples // nb_classes))
        train_labels = np_utils.to_categorical(train_labels_not_encoded)
        train_labels = train_labels[0:train_data_len]

        validation_data = np.load(open(bottleneck_features_path + '/bottleneck_features_validation.npy', 'rb'))
        validation_data_len = len(validation_data)
        validation_labels_not_encoded = np.array(
            [0] * (nb_validation_samples // nb_classes) +
            [1] * (nb_validation_samples // nb_classes) +
            [2] * (nb_validation_samples // nb_classes) +
            [3] * (nb_validation_samples // nb_classes) +
            [4] * (nb_validation_samples // nb_classes) +
            [5] * (nb_validation_samples // nb_classes))
        validation_labels = np_utils.to_categorical(validation_labels_not_encoded)
        validation_labels = validation_labels[0:validation_data_len]

        nodes = 10
        lr = 0.0001

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(nodes, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))

        # model.compile(optimizer='rmsprop',
        #                 loss='categorical_crossentropy', metrics=['accuracy'])

        # optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = optimizers.Adam(lr=lr)


        optimizer_name = 'Adam-lr-{}'.format(lr)
        log_param = 'node-{}-opt-{}'.format(nodes, optimizer_name)
        tbCallBack = callbacks.TensorBoard(log_dir=Path(log_name, log_param).as_posix())

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels),
                  callbacks = [tbCallBack])
        model.save_weights(top_model_weights_path)
        model.save(top_model_path)

    train_top_model()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
