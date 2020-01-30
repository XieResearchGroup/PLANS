import os
from datetime import datetime
from functools import partial

import tensorflow as tf
import numpy as np

from ..models.hmlc import HMLC_L
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec, hierarchical, convert2hier
from ..utils.label_convertors import fill_unlabeled
from .train_model import train_model, noisy_student
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
         comment=None,
         if_hard=False,
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
    x_pred = convert2vec(data_pred[:, 0], dtype=float)

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
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        es_patience=es_patience,
        log_path=log_path,
        log_fh=log_f,
        comment=comment
    )

    # Test label all unlabeled with 1.0 (active)
    log_f.write("Label all of the unlabeled data with 1.0. Train an HMLC_L "
                "model with artifically labeled data in combine with labeled "
                "data.\n")
    # - Initialize model1
    model1 = HMLC_L(drop_rate=drop_rate)
    # - Combine labeled and artificially labeled unlabeled training data
    y_pred = fill_unlabeled(1.0, data_pred[:, 1])
    y_pred = hierarchical(y_pred)
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)
    # - Training
    my_train_model(
        model=model1,
        x_train=x_mix,
        y_train=y_mix,
        unlabeled_weight=1.0,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # Test label all unlabeled with 0.0 (active)
    log_f.write("Label all of the unlabeled data with 0.0. Train an HMLC_L "
                "model with artifically labeled data in combine with labeled "
                "data.\n")
    tf.keras.backend.clear_session()
    model2 = HMLC_L(drop_rate=drop_rate)
    # - Combine labeled and artificially labeled unlabeled training data
    y_pred = fill_unlabeled(0.0, data_pred[:, 1])
    y_pred = hierarchical(y_pred)
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)
    # - Training
    my_train_model(
        model=model2,
        x_train=x_mix,
        y_train=y_mix,
        unlabeled_weight=1.0,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # Train model3
    log_f.write("Train HMLC_L model with Noisy Student with unlabeled weight "
                "set to 0.5\n")
    noisy_student(
        x_train=x_train,
        y_train=y_train,
        unlabeled_weight=unlabeled_weight,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval,
        data_pred=data_pred,
        learning_rate=learning_rate,
        drop_rate=drop_rate,
        batch_size=batch_size,
        epochs=epochs,
        es_patience=es_patience,
        log_path=log_path,
        log_fh=log_f,
        comment=comment)

    log_f.close()


if __name__ == "__main__":
    parser = TrainingArgs()
    args = parser.parse_args()
    main(**vars(args))
