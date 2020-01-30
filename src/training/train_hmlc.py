import os
from datetime import datetime
from functools import partial

import tensorflow as tf
import numpy as np

from models.hmlc import HMLC, HMLC_M, HMLC_L
from data_loaders.cvs_loader import CVSLoader
from utils.label_convertors import convert2vec, hierarchical, convert2hier
from utils.label_convertors import fill_unlabeled
from .train_model import train_model
from .training_args import TrainingArgs


def main(data_path,
         DataLoader=CVSLoader,
         columns=["ECFP", "Label"],
         learning_rate=0.001,
         drop_rate=0.3,
         batch_size=128,
         epochs=30,
         es_patience=5,
         log_path="../logs",
         if_hard=False,
         comment=None,
         unlabeled_weight=0.5):
    # Data
    data_loader = DataLoader(data_path)
    x_train, y_train, x_test, y_test = data_loader.load_data(columns)

    x_train = convert2vec(x_train, dtype=float)
    y_train = convert2hier(y_train, dtype=float)

    x_test = convert2vec(x_test, dtype=float)
    y_val = convert2hier(y_test, dtype=float)  # for validation during training
    y_eval = convert2vec(y_test, dtype=int)    # for evaluation after training

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

    # Set up the train_model function
    my_train_model = partial(
        train_model,
        unlabeled_weight=unlabeled_weight,
        learning_rate=learning_rate,
        drop_rate=drop_rate,
        batch_size=batch_size,
        epochs=epochs,
        es_patience=es_patience,
        log_path=log_path,
        log_fh=log_f,
        comment=comment)

    # Train model1
    # - Initialize model1
    model1 = HMLC(drop_rate=drop_rate)
    # - Training
    my_train_model(
        model=model1,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model1
    predictions = model1.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1], hard_label=if_hard)
    y_pred = hierarchical(y_pred)

    # - Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # Train model2
    tf.keras.backend.clear_session()
    model2 = HMLC_M(drop_rate=drop_rate)
    # - Training
    my_train_model(
        model=model2,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model2
    predictions = model2.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1], hard_label=if_hard)
    y_pred = hierarchical(y_pred)

    # - Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # Train model3
    tf.keras.backend.clear_session()
    model3 = HMLC_L(drop_rate=drop_rate)
    # - Training
    my_train_model(
        model=model3,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    log_f.close()


if __name__ == "__main__":
    parser = TrainingArgs()
    args = parser.parse_args()
    main(**vars(args))
