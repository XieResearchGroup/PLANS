""" Experiment to test HLMC_M and HLMC_L model without using Noisy Student 
labeled unlabeled data.
"""
import os
import argparse
from datetime import datetime
from functools import partial

import tensorflow as tf
import numpy as np

from models.hmlc import HMLC_M, HMLC_L
from data_loaders.cvs_loader import CVSLoader
from utils.label_convertors import convert2vec, hierarchical, convert2hier
from .train_model import train_model
from .training_args import TrainingArgs


def experiment(data_path,
               DataLoader=CVSLoader,
               columns=["ECFP", "Label"],
               learning_rate=0.001,
               drop_rate=0.3,
               batch_size=128,
               epochs=30,
               es_patience=5,
               log_path="../logs",
               comment=None):
    # Data
    data_loader = DataLoader(data_path)
    x_train, y_train, x_test, y_test = data_loader.load_data(columns)

    x_train = convert2vec(x_train, dtype=float)
    y_train = convert2hier(y_train, dtype=float)
    
    x_test = convert2vec(x_test, dtype=float)
    y_val = convert2hier(y_test, dtype=float) # for validation during training
    y_eval = convert2vec(y_test, dtype=int)   # for evaluation after training
    
    data_pred = data_loader.load_unlabeled(["ECFP", "Label"])
    x_pred = data_pred[:, 0]
    x_pred = convert2vec(x_pred, dtype=float)

    # Set up the train_model function
    my_train_model = partial(
        train_model,
        learning_rate=learning_rate,
        drop_rate=drop_rate,
        batch_size=batch_size,
        epochs=epochs,
        es_patience=es_patience,
        log_path=log_path,
        log_fh=log_f,
        comment=comment
    )

    # Open log
    now = datetime.now()
    timestamp = now.strftime(r"%Y%m%d_%H%M%S")
    log_path = os.path.sep.join(log_path.split("/"))
    log_path = os.path.join(log_path, timestamp)
    os.makedirs(log_path, exist_ok=True)
    log_f_path = os.path.join(log_path, "logs.txt")
    log_f = open(log_f_path, "w")

    # Train model1
    ## Initialize model1
    model1 = HMLC_M(drop_rate=drop_rate)
    ## Training
    my_train_model(model1, x_train, y_train, x_test, y_val, y_eval)

    # Train model2
    ## Initialize model2
    tf.keras.backend.clear_session()
    model2 = HMLC_L(drop_rate=drop_rate)
    ## Training
    my_train_model(model2, x_train, y_train, x_test, y_val, y_eval)
    

if __name__ == "__main__":
    parser = TrainingArgs()
    args = parser.parse_args()
    main(**vars(args))
