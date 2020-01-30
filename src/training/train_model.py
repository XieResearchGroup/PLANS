from functools import partial

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
import numpy as np
from sklearn.metrics import accuracy_score

from ..utils.label_convertors import convert2vec, hierarchical, fill_unlabeled
from ..models.hmlc import HMLC, HMLC_M, HMLC_L


def train_model(model,
                x_train,
                y_train,
                unlabeled_weight,
                x_test,
                y_val,
                y_eval,
                learning_rate,
                batch_size,
                epochs,
                es_patience,
                log_path,
                log_fh,
                hier_vio_coef=0.1,
                comment=None):
    """ Train model with provided training and validation data
    """
    # Write comment to the log file
    if comment:
        log_fh.write(comment+"\n")
    # Optimizer
    adam = Adam(learning_rate)
    # Metrics
    precision = Precision()
    recall = Recall()
    auc = AUC()
    accuracy = Accuracy()
    # Compile model
    model.compile(
        optimizer=adam,
        loss=model.training_loss(unlabeled_weight, hier_vio_coef),
        metrics=[accuracy, precision, recall, auc]
    )

    # Model training
    # - Callbacks
    tbcb = TensorBoard(log_path)
    escb = EarlyStopping(
        "val_loss",
        patience=es_patience,
        restore_best_weights=True
    )
    # - fit
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
    log_fh.write("acc_score is: {}\n".format(acc_score))


def noisy_student(x_train,
                  y_train,
                  unlabeled_weight,
                  x_test,
                  y_val,
                  y_eval,
                  data_pred,
                  learning_rate,
                  drop_rate,
                  batch_size,
                  epochs,
                  es_patience,
                  log_path,
                  log_fh,
                  comment):

    noisy_train_model = partial(
        train_model,
        unlabeled_weight=unlabeled_weight,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        es_patience=es_patience,
        log_path=log_path,
        log_fh=log_fh,
        comment=comment)

    # Train model3
    log_fh.write("Noisy Student model 1:\n")
    tf.keras.backend.clear_session()
    model1 = HMLC(drop_rate=drop_rate)
    # - train model3_1
    noisy_train_model(
        model=model1,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model3_1
    x_pred = convert2vec(data_pred[:, 0], dtype=float)
    predictions = model1.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1], hard_label=False)
    y_pred = hierarchical(y_pred)

    # - Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # - Train model3_2
    tf.keras.backend.clear_session()
    model2 = HMLC_M(drop_rate=drop_rate)
    # - Training
    noisy_train_model(
        model=model2,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model3_2
    predictions = model2.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1], hard_label=False)
    y_pred = hierarchical(y_pred)

    # - Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # - Train model3_3
    tf.keras.backend.clear_session()
    model3 = HMLC_L(drop_rate=drop_rate)
    # - Training
    noisy_train_model(
        model=model3,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)
