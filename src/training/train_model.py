import os
from functools import partial

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from ..utils.label_convertors import convert2vec, hierarchical, fill_unlabeled
from ..utils.training_utils import training_log
from ..utils.mixup import mixup
from ..models.hmlc import HMLC, HMLC_M, HMLC_L, HMLC_XL, HMLC_XXL


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
        restore_best_weights=True,
        mode="min"
    )
    lrcb = LearningRateScheduler(model.scheduler)
    # - fit
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tbcb, escb, lrcb],
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

    return model


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
                  comment,
                  mixup_=None,
                  repeat=None):

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

    # Train model1
    log_fh.write("Noisy Student model 1:\n")
    tf.keras.backend.clear_session()
    model1 = HMLC(drop_rate=drop_rate)
    if mixup_:
        x_train, y_train = mixup(
            mixup_, mixup_, x_train, y_train, repeat=repeat, shuffle=True)
    # - train model1
    model1 = noisy_train_model(
        model=model1,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model1
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

    # - Train model2
    tf.keras.backend.clear_session()
    log_fh.write("Noisy Student model 2:\n")
    model2 = HMLC_M(drop_rate=drop_rate)
    if mixup_:
        x_mix, y_mix = mixup(
            mixup_, mixup_, x_mix, y_mix, repeat=repeat, shuffle=True)
    # - Training
    model2 = noisy_train_model(
        model=model2,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model2
    predictions = model2.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1], hard_label=False)
    y_pred = hierarchical(y_pred)

    # - Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # - Train model3
    tf.keras.backend.clear_session()
    log_fh.write("Noisy Student model 3:\n")
    model3 = HMLC_L(drop_rate=drop_rate)
    if mixup_:
        x_mix, y_mix = mixup(
            mixup_, mixup_, x_mix, y_mix, repeat=repeat, shuffle=True)
    # - Training
    model3 = noisy_train_model(
        model=model3,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    return model3


def noisy_student_2(x_train,
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
    r""" Train HLMC with Noisy Student
    The difference from noisy_student is this method utilize the partially
    labeled data in the CYP450 training set.
    """

    noisy_train_model = partial(
        train_model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        es_patience=es_patience,
        log_path=log_path,
        log_fh=log_fh,
        comment=comment)

    # Train model1
    log_fh.write("Noisy Student model 1 (utilized partially labeled data):\n")
    tf.keras.backend.clear_session()
    model1 = HMLC(drop_rate=drop_rate)
    # - Use partially labeled data as the input of model1
    x_pred = convert2vec(data_pred[:, 0], dtype=float)
    y_pred = fill_unlabeled(0.5, data_pred[:, 1])
    y_pred = hierarchical(y_pred)

    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)
    # - train model1
    model1 = noisy_train_model(
        model=model1,
        x_train=x_mix,
        y_train=y_mix,
        unlabeled_weight=0.,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model1
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

    # - Train model2
    tf.keras.backend.clear_session()
    log_fh.write("Noisy Student model 2:\n")
    model2 = HMLC_M(drop_rate=drop_rate)
    # - Training
    model2 = noisy_train_model(
        model=model2,
        x_train=x_mix,
        y_train=y_mix,
        unlabeled_weight=unlabeled_weight,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model2
    predictions = model2.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1], hard_label=False)
    y_pred = hierarchical(y_pred)

    # - Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # - Train model3
    tf.keras.backend.clear_session()
    log_fh.write("Noisy Student model 3:\n")
    model3 = HMLC_L(drop_rate=drop_rate)
    # - Training
    model3 = noisy_train_model(
        model=model3,
        x_train=x_mix,
        y_train=y_mix,
        unlabeled_weight=unlabeled_weight,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    return model3


def noisy_student_L(x_train,
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

    model0 = noisy_student(x_train,
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
                           comment)

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

    # - Predict labels for unlabeled data with model0
    x_pred = convert2vec(data_pred[:, 0], dtype=float)
    predictions = model0.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1], hard_label=False)
    y_pred = hierarchical(y_pred)

    # - Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # - Train model1
    tf.keras.backend.clear_session()
    log_fh.write("Noisy Student larger model 1:\n")
    model1 = HMLC_XL(drop_rate=drop_rate)
    # - Training
    model1 = noisy_train_model(
        model=model1,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    # - Predict labels for unlabeled data with model1
    predictions = model1.predict(x_pred)[:, -5:]
    y_pred = fill_unlabeled(predictions, data_pred[:, 1], hard_label=False)
    y_pred = hierarchical(y_pred)

    # - Combine labeled and unlabeled training data
    x_mix = np.concatenate([x_train, x_pred], axis=0)
    y_mix = np.concatenate([y_train, y_pred], axis=0)
    randomed_idx = np.random.permutation(x_mix.shape[0])
    np.take(x_mix, randomed_idx, axis=0, out=x_mix)
    np.take(y_mix, randomed_idx, axis=0, out=y_mix)

    # - Train model2
    tf.keras.backend.clear_session()
    log_fh.write("Noisy Student larger model 2:\n")
    model2 = HMLC_XXL(drop_rate=drop_rate)
    # - Training
    model2 = noisy_train_model(
        model=model2,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval)

    return model2


def predict_and_mix(model, x_pred, x, y, shuffle=True):
    r""" Make predictions with the given model and mix it with the existing
    training set.
    model (tf.keras.Model): the pretrained model to make predictions
    x_pred (np.array): inputs for the model
    x (np.arrapy): training set data
    y (np.array): training set label
    shuffle (bool): if shuffle after mixing the training and predicted data
    ===========================================================================
    return:
    x_mix: mixed training data
    y_mix: mixed lables (soft)
    """
    y_pred = model.predict(x_pred)
    x_mix = np.concatenate([x, x_pred], axis=0)
    y_mix = np.concatenate([y, y_pred], axis=0)
    if shuffle:
        randomed_idx = np.random.permutation(x_mix.shape[0])
        np.take(x_mix, randomed_idx, axis=0, out=x_mix)
        np.take(y_mix, randomed_idx, axis=0, out=y_mix)
    return x_mix, y_mix


def predict_with_multiteacher_and_mix(teachers, x_pred, x, y, shuffle=True):
    r""" Make predictions with multiple teacher models. Each teacher predicts
    the label for one class only.
    model (list): a list of pretrained models to make predictions
    x_pred (np.array): inputs for the model
    x (np.arrapy): training set data
    y (np.array): training set label
    shuffle (bool): if shuffle after mixing the training and predicted data
    ===========================================================================
    return:
    x_mix: mixed training data
    y_mix: mixed lables (soft)
    """
    y_pred = np.zeros((x_pred.shape[0], len(teachers)))
    for i, model in enumerate(teachers):
        y_pred[:, i] = model.predict(x_pred)
    x_mix = np.concatenate([x, x_pred], axis=0)
    y_mix = np.concatenate([y, y_pred], axis=0)
    if shuffle:
        randomed_idx = np.random.permutation(x_mix.shape[0])
        np.take(x_mix, randomed_idx, axis=0, out=x_mix)
        np.take(y_mix, randomed_idx, axis=0, out=y_mix)
    return x_mix, y_mix


def ns_linear_teacher_model(model,
                            x_train,
                            y_train,
                            x_test,
                            y_test,
                            x_pred,
                            batch_size,
                            epochs,
                            cb_list,
                            log_f,
                            log_path,
                            n_repeat=3,
                            mixup_coef=None,
                            mixup_repeat=None):
    r""" Train linear model with Noisy Student
    model: the model to be trained
    x_train: training data
    y_train: labels of the training data
    x_pred: unlabeled training data
    cb_list: callback list
    log_f: logging file handler
    log_path: path to the logging directory
    n_repeat: times to train the model
    ===========================================================================
    return: the trained model
    """
    histories = list()
    log_f.write("training {}:\n".format(str(model)))
    train_his = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=cb_list,
        validation_data=[x_test, y_test]
    )
    histories.append(train_his)

    y_pred = model.predict(x_test)
    training_log(train_his, y_pred, y_test, log_f)

    # repeat training the model
    for i in range(n_repeat):
        log_f.write(
            "repeat training {}, {}/{}:\n".format(str(model), i+1, n_repeat))
        # label unlabled
        x_mix, y_mix = predict_and_mix(
            model=model,
            x_pred=x_pred,
            x=x_train,
            y=y_train,
            shuffle=True
        )
        if mixup_coef is not None:
            x_mix, y_mix = mixup(mixup_coef, mixup_coef, x_mix, y_mix,
                                 repeat=mixup_repeat, shuffle=True)
        # train model with the mixed data
        train_his = model.fit(
            x=x_mix,
            y=y_mix,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=cb_list,
            validation_data=[x_test, y_test]
        )
        histories.append(train_his)
        # log training history
        y_pred = model.predict(x_test)
        training_log(train_his, y_pred, y_test, log_f)

    return model, histories


def ns_linear_student_model(model,
                            x_train,
                            y_train,
                            x_mix,
                            y_mix,
                            x_test,
                            y_test,
                            x_pred,
                            batch_size,
                            epochs,
                            cb_list,
                            log_f,
                            log_path,
                            n_repeat=3,
                            mixup_coef=None,
                            mixup_repeat=None):
    r""" Train student linear model with Noisy Student
    model: the model to be trained
    x_train: training data
    y_train: labels of the training data
    x_mix: the input data used by the teacher model
    y_mix: labeled data generated by the teacher model
    x_pred: unlabeled training data
    cb_list: callback list
    log_f: logging file handler
    log_path: path to the logging directory
    n_repeat: times to train the model
    ===========================================================================
    return: the trained model
    """
    histories = list()
    log_f.write("training {}:\n".format(str(model)))
    train_his = model.fit(
        x=x_mix,
        y=y_mix,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=cb_list,
        validation_data=[x_test, y_test]
    )
    histories.append(train_his)

    y_pred = model.predict(x_test)
    training_log(train_his, y_pred, y_test, log_f)

    # repeat training the model
    for i in range(n_repeat):
        log_f.write(
            "repeat training {}, {}/{}:\n".format(str(model), i+1, n_repeat))
        # label unlabled
        x_mix, y_mix = predict_and_mix(
            model=model,
            x_pred=x_pred,
            x=x_train,
            y=y_train,
            shuffle=True
        )
        if mixup_coef is not None:
            x_mix, y_mix = mixup(mixup_coef, mixup_coef, x_mix, y_mix,
                                 repeat=mixup_repeat, shuffle=True)
        # train model with the mixed data
        train_his = model.fit(
            x=x_mix,
            y=y_mix,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=cb_list,
            validation_data=[x_test, y_test]
        )
        histories.append(train_his)
        # log training history
        y_pred = model.predict(x_test)
        training_log(train_his, y_pred, y_test, log_f)

    return model, histories


def plot_history(train_his, path, prefix):
    r""" Plot the training history
    train_his: Tensorflow History callback object
    path: path to save the ploting
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[12.8, 4.8])
    x_max = 0
    y_loss_max = 0
    y_acc_max = 0
    for i, his in enumerate(train_his):
        train_loss = his.history["loss"]
        val_loss = his.history["val_loss"]
        train_acc = his.history["acc"]
        val_acc = his.history["val_acc"]
        axes[0].plot(
            list(range(len(train_loss))),
            train_loss,
            label="train_loss_"+str(i)
        )
        axes[0].plot(
            list(range(len(val_loss))), val_loss, label="val_loss_"+str(i))
        axes[1].plot(
            list(range(len(train_acc))), train_acc, label="train_acc_"+str(i))
        axes[1].plot(
            list(range(len(val_acc))), val_acc, label="val_acc_"+str(i))
        for data in [train_loss, val_loss]:
            x_max = max(x_max, len(data))
            y_loss_max = max(y_loss_max, max(data))
        for data in [train_acc, val_acc]:
            x_max = max(x_max, len(data))
            y_acc_max = max(y_acc_max, max(data))

    lgds = list()
    for ax in axes:
        ax.set(xlim=[0, x_max+1])
        handles, labels = ax.get_legend_handles_labels()
        handles = [h for _, h in sorted(zip(labels, handles))]
        labels.sort()
        lgd = ax.legend(handles, labels,
                        bbox_to_anchor=(0., -0.1, 1, 0),
                        loc=9, ncol=2, mode=None, borderaxespad=0.)
        lgds.append(lgd)

    axes[0].set(ylim=[0, y_loss_max+0.1], title="loss")
    axes[1].set(ylim=[0, y_acc_max+0.1], title="accuracy")

    plt.savefig(path+os.path.sep+prefix+"_training_results.png",
                bbox_extra_artists=lgds,
                bbox_inches='tight')
