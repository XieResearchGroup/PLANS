import os
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
import numpy as np

from models.hmlc import HMLC
from data_loaders.cvs_loader import CVSLoader


def convert2vec(data, dtype=int):
    data = data.tolist()
    data = list(map(list, data))
    data = [list(map(dtype, d)) for d in data]
    data = np.array(data)
    return data


def _if_true_label(label):
    try:
        return np.expand_dims((np.sum(label, axis=1)>0).astype(np.int32), 1)
    except np.AxisError:
        return np.expand_dims(((label>0).astype(np.int32)), 1)


def hierarchical(true_label):
    l1 = _if_true_label(true_label)
    l2_1 = _if_true_label(true_label[:, 0:2])
    l2_2 = _if_true_label(
        true_label[:, 2:4])
    l2_3 = _if_true_label(true_label[:, 4])
    return np.concatenate(
        [l1, l2_1, l2_2, l2_3, true_label, true_label], axis=1)


def main(data_path,
         learning_rate=0.001,
         drop_rate=0.3,
         batch_size=128,
         epochs=30):
    # Initialize model
    model = HMLC(drop_rate=drop_rate)
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
    ## Data
    data_loader = CVSLoader(data_path)
    x_train, y_train, x_test, y_test = data_loader.load_data(["ECFP", "Label"])
    x_train = convert2vec(x_train, dtype=float)
    y_train = convert2vec(y_train, dtype=int)
    y_train = hierarchical(y_train)
    y_train = y_train.astype(float)
    x_test = convert2vec(x_test, dtype=float)
    y_test = convert2vec(y_test, dtype=int)
    # y_test = hierarchical(y_test)
    # y_test = y_test.astype(float)
    ## Callbacks
    now = datetime.now()
    timestamp = now.strftime(r"%Y%m%d_%H%M%S")
    tbcb = TensorBoard(os.path.join("..", "logs", timestamp))
    ## fit
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tbcb]
        # validation_data=[x_test, y_test]
    )

    # Predict labels for unlabeled data
    data_pred = data_loader.load_unlabeled(["ECFP", "Label"])
    x_pred = data_pred[:, 0]
    x_pred = convert2vec(x_pred, dtype=float)
    predictions = model.predict(x_pred)
    log_path = os.path.join("..", "logs", timestamp, "predictions.txt")
    with open(log_path, "w") as f:
        np.savetxt(f, predictions)
        np.savetxt(f, data_pred[:, 1], fmt="%s")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data-path")
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-l", "--learning-rate", type=float)
    parser.add_argument("-e", "--epochs", type=int)
    args = parser.parse_args()
    main(**vars(args))
