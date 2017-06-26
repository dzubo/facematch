Facematch
==============================

## The project
The CNN used to identify emotions and help people to learn accurate facial expressions.  
[Hackathon Presentation](https://docs.google.com/presentation/d/1RXmRfiQj4Uh1Q2nfpQoh98v15LFccNStylFzlv_rV_U/edit?usp=sharing)  
This repository contains dataset visualization conde and deep learning training code.  
You can find server and ios code here: https://github.com/kkwoker/facematch

## Issues
People lack in Emotional Quotient and most of the games do not include the emotional feedback or input; so that they can improve their EQ.

## Solution
FaceMatch takes facial expressions as the input from the users and gives score based on the accuracy of their emotions. It is based on the Deep Neural Network which is trained on about 10,000 images. The prototype for FaceMatch is an iOS app.

## How It Works?
![steps diagram](/reports/figures/how-it-works.png)

--------------------------------------------------

## Emotion recognition from images

Download the [fer2013.csv file](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

Extract images from the fer2013.csv file:
```
python src/data/make_dataset.py data/raw/fer2013/fer2013.csv data/processed/fer2013/all/
```

Generate bottleneck features on VGG16 net:

```
python src/features/build_features.py data/processed/fer2013/all/ data/interim/
```

Train the model:

```
python src/models/train_model.py data/interim/ models/fer2013-01/ data/logs/012
```

## TODO

1. Balance training and validation datasets
1. Do data augmentation
1. Ensure the images cycle over epochs
1. Use color dataset

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
