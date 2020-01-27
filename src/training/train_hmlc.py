import os
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
import numpy as np
from sklearn.metrics import accuracy_score

from models.hmlc import HMLC, HMLC_M, HMLC_L
from data_loaders.cvs_loader import CVSLoader
from utils.label_convertors import convert2vec, hierarchical, convert2hier


def train_model(model, 
                x_train,
                y_train,
                x_test,
                y_val,
                y_eval,
                learning_rate,
                drop_rate,
                batch_size,
                epochs,
                log_path,
                log_fh=None):
    ## Optimizer
    adam = Adam(learning_rate)
    ## Metrics
    precision = Precision()
    recall = Recall()
    auc = AUC()
    accuracy = Accuracy()
    ## Compile model
    model.compile(
        optimizer=adam,
        loss=model.training_loss,
        metrics=[accuracy, precision, recall, auc]
    )

    # Model training
    ## Callbacks
    tbcb = TensorBoard(log_path)
    escb = EarlyStopping("val_accuracy", patience=5)
    ## fit
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tbcb, escb],
        validation_data=[x_test, y_val]
    )

    # Model evaluation
    evaluation = np.round(model.predict(x_test))[:, -5:]
    acc_score = accuracy_score(
        np.squeeze(y_eval.reshape(1, -1)),
        np.squeeze(evaluation.reshape(1, -1)))

    # Show and save training results
    print("acc_score is: {}".format(acc_score))
    if log_fh is not None:
        print("acc_score is: {}".format(acc_score), file=log_fh)


def fill_unlabeled(predictions, data_unlabeled):
    """ Fill the unlabeled blanks in data_unlabeled with predicted labels
    predictions (numpy.array): predicted labels, shape is (?, 5)
    data_unlabeled (numpy.array): str, unlabeled data in "1_10_"-like format
    ========================================================================
    return: numpy.array
    """
    data_labeled = np.zeros(predictions.shape)
    for i, data in enumerate(data_unlabeled):
        labeled = list(data)
        for j, label in enumerate(labeled):
            try:
                labeled[j] = int(label)
            except ValueError:
                labeled[j] = predictions[i, j]
        data_labeled[i] = labeled
    return data_labeled


def main(data_path,
         DataLoader=CVSLoader,
         columns=["ECFP", "Label"],
         learning_rate=0.001,
         drop_rate=0.3,
         batch_size=128,
         epochs=30,
         log_path="../logs"):
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
    y_pred = fill_unlabeled(predictions, data_pred[:, 1])
    y_pred = convert2hier(np.round(y_pred), dtype=float)

    ## Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # Train model2
    model2 = HMLC_M(drop_rate=drop_rate)
    ## Training
    train_model(
        model2, x_mix, y_mix, x_test, y_val, y_eval,
        learning_rate, drop_rate, batch_size, epochs, log_path, log_f
    )

    ## Predict labels for unlabeled data with model2
    predictions = model2.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1])
    y_pred = convert2hier(np.round(y_pred), dtype=float)

    ## Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # Train model3
    model3 = HMLC_L(drop_rate=drop_rate)
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
    args = parser.parse_args()
    main(**vars(args))
