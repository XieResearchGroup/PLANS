import os
import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np

from models.hmlc import HMLC, HMLC_M, HMLC_L
from data_loaders.cvs_loader import CVSLoader
from utils.label_convertors import convert2vec, hierarchical, convert2hier
from utils.label_convertors import fill_unlabeled
from .train_model import train_model


def main(data_path,
         DataLoader=CVSLoader,
         columns=["ECFP", "Label"],
         learning_rate=0.001,
         drop_rate=0.3,
         batch_size=128,
         epochs=30,
         log_path="../logs",
         if_hard=False):
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
    model1 = HMLC(drop_rate=drop_rate)
    ## Training
    train_model(
        model1, x_train, y_train, x_test, y_val, y_eval,
        learning_rate, drop_rate, batch_size, epochs, log_path, log_f
    )

    ## Predict labels for unlabeled data with model1
    predictions = model1.predict(x_pred)[:, -5:]
    y_pred_soft = fill_unlabeled(
        predictions, data_pred[:, 1], hard_label=False)
    y_pred_soft = hierarchical(y_pred_soft)

    y_pred_hard = fill_unlabeled(predictions, data_pred[:, 1], hard_label=True)
    y_pred_hard = hierarchical(y_pred_hard)

    ## Combine labeled and soft-labeled unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred_soft], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # Train model2 with soft labels
    model2 = HMLC_M(drop_rate=drop_rate)
    ## Training
    train_model(
        model2, x_mix, y_mix, x_test, y_val, y_eval,
        learning_rate, drop_rate, batch_size, epochs, log_path, log_f
    )

    ## Combine labeled and hard-labeled unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred_hard], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # Train model3 with hard labels
    model3 = HMLC_M(drop_rate=drop_rate)
    ## Training
    train_model(
        model3, x_mix, y_mix, x_test, y_val, y_eval,
        learning_rate, drop_rate, batch_size, epochs, log_path, log_f
    )
    log_f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data-path")
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-l", "--learning-rate", type=float)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-r", "--if-hard", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
