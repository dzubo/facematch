# -*- coding: utf-8 -*-
import os
from pathlib import Path
import shutil
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np

from PIL import Image


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    p = Path(input_filepath)
    data = pd.read_csv(p)

    images = data.pixels
    labels = data.emotion
    images = images.apply(lambda x: x.split(' '))
    images = list(images)
    images = np.asfarray(images)
    emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    # classes = [v for v in emotions.values()]


    # labels = [child.name for child in p.iterdir() if child.is_dir()]
    # print(labels)

    im_number = images.shape[1]
    train_number = round(im_number * 0.8)
    print('Image number: {}'.format(im_number))

    for i in range(0, im_number):
        print(i)
        img = images[i].reshape(48, 48)

        im = Image.fromarray(img).convert('RGB')
        im_name = '{:04d}'.format(i) + '.jpg'
        data_type = 'train'
        if i > train_number:
            data_type = 'validation'

        im_path = Path(output_filepath, data_type, emotions[labels[i]])
        im_path.mkdir(parents=True, exist_ok=True)

        im.save(Path(im_path, im_name))

    # process files, copy to 'test' and 'validate' folders
    # for label in labels:
    #     p = Path(input_filepath, label)
    #     files = list(p.iterdir())
    #     files_num = len(files)

    #     # output some stats
    #     print('\'{}\': {} photo(s)'.format(label, files_num))

    #     # create output folders for the label
    #     train_path = Path(output_filepath, 'train', label)
    #     train_path.mkdir(parents=True, exist_ok=True)
    #     validate_path = Path(output_filepath, 'validation', label)
    #     validate_path.mkdir(parents=True, exist_ok=True)

    #     for f in files[0:40]:
    #         shutil.copy(f.resolve(), train_path)
    #     for f in files[40:]:
    #         shutil.copy(f.resolve(), validate_path)

    # create output folders

    logger = logging.getLogger(__name__)
    logger.info('making images set from FER2013 csv file')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
