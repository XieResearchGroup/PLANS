import os

import tensorflow as tf
from tensorflow import math
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Add, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam


class HMLC(Model):

    def __init__(self,
                 fp_len=2048,
                 l1_len=1,
                 l2_len=3,
                 l3_len=5,
                 beta=0.5,
                 drop_rate=0.3):
        super(HMLC, self).__init__()
        self.beta = beta
        self.fp_len = fp_len
        self.l1_len = l1_len
        self.l2_len = l2_len
        self.l3_len = l3_len
        self.global_dense1 = Dense(fp_len, activation="elu")
        self.global_dense2 = Dense(fp_len, activation="elu")
        self.global_dense3 = Dense(fp_len, activation="elu")
        self.global_dense4 = Dense(l3_len, activation=None)
        self.local_dense1 = Dense(l1_len, activation=None)
        self.local_dense2 = Dense(l2_len, activation=None)
        self.local_dense3 = Dense(l3_len, activation=None)
        self.dropout = Dropout(drop_rate)
        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()
        self.add4 = Add()

    def call(self, inputs, training=None):
        out_g1 = self.global_dense1(inputs)
        out_l1 = self.local_dense1(inputs)

        in_g2 = self.add1([inputs, out_g1])

        out_g2 = self.global_dense2(in_g2)
        out_l2 = self.local_dense2(in_g2)

        in_g3 = self.add2([inputs, out_g2])

        out_g3 = self.global_dense3(in_g3)
        out_l3 = self.local_dense3(in_g3)

        in_g4 = self.add3([inputs, out_g3])
        in_g4 = self.dropout(in_g4)

        out_g4 = self.global_dense4(in_g4)
        out_global = math.multiply(self.beta, out_g4)
        out_local = math.multiply(1-self.beta, out_l3)

        global_local_sum = self.add4([out_global, out_local])
        
        final_out = tf.sigmoid(tf.concat(
            [out_l1, out_l2, out_l3, global_local_sum], axis=1))
        return final_out

    def training_loss(self, y_true, y_pred, hier_vio_coef=0.1):
        cl = self.crossentropy_loss(y_true, y_pred)
        hv = self.hierarchical_violation(
            y_true, y_pred, hier_vio_coef)
        return tf.add(cl, hv)

    def crossentropy_loss(self, y_true, y_pred, weights=None):
        global_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_true[:, -self.l3_len:],
                y_pred[:, -self.l3_len:]))
        local_loss_1 = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_true[:, 0:self.l1_len],
                y_pred[:, 0:self.l1_len]))
        local_loss_2 = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_true[:, self.l1_len:self.l1_len+self.l2_len],
                y_pred[:, self.l1_len:self.l1_len+self.l2_len]))
        local_loss = tf.add(local_loss_1, local_loss_2)
        return tf.add(global_loss, local_loss)
    
    def hierarchical_violation(self, y_true, y_pred, hier_vio_coef):
        l1 = tf.nn.sigmoid(y_pred[:, 0:self.l1_len])
        l2 = tf.nn.sigmoid(y_pred[:, self.l1_len:self.l1_len+self.l2_len])
        l3 = tf.nn.sigmoid(y_pred[:, self.l1_len+self.l2_len:])

        l1_l2 = tf.reduce_sum(tf.maximum(0.0, tf.subtract(l2, l1)))
        l2_l3_1 = tf.reduce_sum(
            tf.maximum(
                0.0, tf.subtract(l3[:, 0:2], tf.expand_dims(l2[:, 0], 1))))
        l2_l3_2 = tf.reduce_sum(
            tf.maximum(
                0.0, tf.subtract(l3[:, 2:4], tf.expand_dims(l2[:, 1], 1))))
        l2_l3_3 = tf.reduce_sum(
            tf.maximum(
                0.0, tf.subtract(l3[:, 4], tf.expand_dims(l2[:, 2], 1))))
        hier_viol = tf.multiply(
            hier_vio_coef, (l1_l2 + l2_l3_1 + l2_l3_2 + l2_l3_3))
        return hier_viol


class HMLC_M(HMLC):

    def __init__(self, *args, **kwargs):
        super(HMLC_M, self).__init__(*args, **kwargs)
        self.global_dense4 = Dense(int(self.fp_len/2), activation="elu")
        self.global_dense5 = Dense(self.l3_len, activation=None)
        self.local_dense1_1 = Dense(self.fp_len, activation="elu")
        self.local_dense2_1 = Dense(self.fp_len, activation="elu")
        self.local_dense3_1 = Dense(self.fp_len, activation="elu")

    def call(self, inputs, training=None):
        out_g1 = self.global_dense1(inputs)
        out_l1_1 = self.local_dense1_1(inputs)
        out_l1 = self.local_dense1(out_l1_1)


        in_g2 = self.add1([inputs, out_g1])
        out_g2 = self.global_dense2(in_g2)
        out_l2_1 = self.local_dense2_1(in_g2)
        out_l2 = self.local_dense2(out_l2_1)

        in_g3 = self.add2([inputs, out_g2])
        out_g3 = self.global_dense3(in_g3)
        out_l3_1 = self.local_dense3_1(in_g3)
        out_l3 = self.local_dense3(out_l3_1)

        in_g4 = self.add3([inputs, out_g3])
        out_g4 = self.global_dense4(in_g4)

        in_g5 = self.dropout(out_g4)
        out_g5 = self.global_dense5(in_g5)

        out_global = math.multiply(self.beta, out_g5)
        out_local = math.multiply(1-self.beta, out_l3)

        global_local_sum = self.add4([out_global, out_local])
        
        final_out = tf.sigmoid(tf.concat(
            [out_l1, out_l2, out_l3, global_local_sum], axis=1))
        return final_out


class HMLC_L(HMLC_M):

    def __init__(self, *args, **kwargs):
        super(HMLC_L, self).__init__(*args, **kwargs)
        self.global_dense5 = Dense(int(self.fp_len/4), activation="elu")
        self.global_dense6 = Dense(self.l3_len, activation=None)
        self.local_dense1_2 = Dense(int(self.fp_len/2), activation="elu")
        self.local_dense2_2 = Dense(int(self.fp_len/2), activation="elu")
        self.local_dense3_2 = Dense(int(self.fp_len/2), activation="elu")

    def call(self, inputs, training=None):
        out_g1 = self.global_dense1(inputs)
        out_l1_1 = self.local_dense1_1(inputs)
        out_l1_2 = self.local_dense1_2(out_l1_1)
        out_l1 = self.local_dense1(out_l1_2)


        in_g2 = self.add1([inputs, out_g1])
        out_g2 = self.global_dense2(in_g2)
        out_l2_1 = self.local_dense2_1(in_g2)
        out_l2_2 = self.local_dense2_2(out_l2_1)
        out_l2 = self.local_dense2(out_l2_2)

        in_g3 = self.add2([inputs, out_g2])
        out_g3 = self.global_dense3(in_g3)
        out_l3_1 = self.local_dense3_1(in_g3)
        out_l3_2 = self.local_dense3_2(out_l3_1)
        out_l3 = self.local_dense3(out_l3_2)

        in_g4 = self.add3([inputs, out_g3])
        out_g4 = self.global_dense4(in_g4)

        in_g5 = self.dropout(out_g4)
        out_g5 = self.global_dense5(in_g5)

        in_g6 = self.dropout(out_g5)
        out_g6 = self.global_dense6(in_g6)

        out_global = math.multiply(self.beta, out_g6)
        out_local = math.multiply(1-self.beta, out_l3)

        global_local_sum = self.add4([out_global, out_local])
        
        final_out = tf.sigmoid(tf.concat(
            [out_l1, out_l2, out_l3, global_local_sum], axis=1))
        return final_out
