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


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    # dimensions of our images.
    img_width, img_height = 48, 48

    train_data_dir = input_filepath + '/train/'
    validation_data_dir = input_filepath + '/validation/'
    bottleneck_save_path = output_filepath
    nb_train_samples = 1852
    nb_validation_samples = 468
    batch_size = 16

    def save_bottlebeck_features():
        datagen = ImageDataGenerator(rescale=1. / 255)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')
        print('The VGG16 network weights downloaded')

        generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        print('Calculating bottleneck features, training dataset')
        bottleneck_features_train = model.predict_generator(
            generator, nb_train_samples // batch_size)
        print('Saving bottleneck features, training dataset')
        np.save(open(bottleneck_save_path + '/bottleneck_features_train.npy', 'wb'),
                bottleneck_features_train)

        generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        print('Calculating bottleneck features, validation dataset')
        bottleneck_features_validation = model.predict_generator(
            generator, nb_validation_samples // batch_size)

        print('Saving bottleneck features, validation dataset')
        np.save(open(bottleneck_save_path + '/bottleneck_features_validation.npy', 'wb'),
                bottleneck_features_validation)

    save_bottlebeck_features()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
