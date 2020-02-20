import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Add, Dropout


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
        out_l1 = self.local_dense1(out_g1)

        in_g2 = self.add1([inputs, out_g1])

        out_g2 = self.global_dense2(in_g2)
        out_l2 = self.local_dense2(out_g2)

        in_g3 = self.add2([inputs, out_g2])

        out_g3 = self.global_dense3(in_g3)
        out_l3 = self.local_dense3(out_g3)

        in_g4 = self.add3([inputs, out_g3])
        in_g4 = self.dropout(in_g4)

        out_g4 = self.global_dense4(in_g4)
        out_global = tfm.multiply(self.beta, out_g4)
        out_local = tfm.multiply(1-self.beta, out_l3)

        global_local_sum = self.add4([out_global, out_local])

        final_out = tf.sigmoid(tf.concat(
            [out_l1, out_l2, out_l3, global_local_sum], axis=1))
        return final_out

    def training_loss(self, unlabeled_weight, hier_vio_coef=0.1):

        def loss(y_true, y_pred):
            cl = self.crossentropy_loss(y_true, y_pred, unlabeled_weight)
            hv = self.hierarchical_violation(
                y_true, y_pred, hier_vio_coef)
            return tf.add(cl, hv)
            # return cl

        return loss

    @staticmethod
    def weighted_binary_crossentropy(y_true,
                                     y_pred,
                                     unlabeled_weight,
                                     epsilon=1e-12):
        weights = tf.ones_like(y_true, dtype=tf.float32)
        unlabeled = tf.logical_and(tf.greater(y_true, 0), tf.less(y_true, 1))
        weights = tf.where(unlabeled, float(unlabeled_weight), weights)
        # y_pred = tf.clip_by_value(y_pred, epsilon, 1.-epsilon)
        ce = tf.negative(
            tf.add(
                tf.multiply(y_true, tfm.log(y_pred+epsilon)),
                tf.multiply(
                    tf.subtract(1.0, y_true),
                    tfm.log(tf.subtract(1.0, y_pred)+epsilon))))
        weighted_ce = tf.multiply(weights, ce)
        crossentropy = tf.reduce_mean(weighted_ce)
        return crossentropy

    def crossentropy_loss(self, y_true, y_pred, unlabeled_weight):
        y_true_global = y_true[:, -self.l3_len:]
        y_pred_global = y_pred[:, -self.l3_len:]
        y_true_l1 = y_true[:, 0:self.l1_len]
        y_pred_l1 = y_pred[:, 0:self.l1_len]
        y_true_l2 = y_true[:, self.l1_len:self.l1_len+self.l2_len]
        y_pred_l2 = y_pred[:, self.l1_len:self.l1_len+self.l2_len]

        # global_loss = tf.reduce_mean(
        #     tf.keras.losses.binary_crossentropy(
        #         y_true_global, y_pred_global))
        # local_loss_1 = tf.reduce_mean(
        # tf.keras.losses.binary_crossentropy(y_true_l1, y_pred_l1))
        # local_loss_2 = tf.reduce_mean(
        #     tf.keras.losses.binary_crossentropy(y_true_l2, y_pred_l2))

        global_loss = self.weighted_binary_crossentropy(
            y_true_global, y_pred_global, unlabeled_weight)
        local_loss_1 = self.weighted_binary_crossentropy(
            y_true_l1, y_pred_l1, unlabeled_weight)
        local_loss_2 = self.weighted_binary_crossentropy(
            y_true_l2, y_pred_l2, unlabeled_weight)

        local_loss = tf.add(local_loss_1, local_loss_2)
        return tf.add(global_loss, local_loss)

    def hierarchical_violation(self,
                               y_true,
                               y_pred,
                               hier_vio_coef,
                               epsilon=1e-12):
        l1 = y_pred[:, 0:self.l1_len]
        l2 = y_pred[:, self.l1_len:self.l1_len+self.l2_len]
        l3 = y_pred[:, self.l1_len+self.l2_len:]

        l1_l2 = tf.reduce_mean(tf.maximum(0.0, tf.subtract(l2, l1)))
        l2_l3_1 = tf.reduce_mean(
            tf.maximum(
                0.0, tf.subtract(l3[:, 0:2], tf.expand_dims(l2[:, 0], 1))))
        l2_l3_2 = tf.reduce_mean(
            tf.maximum(
                0.0, tf.subtract(l3[:, 2:4], tf.expand_dims(l2[:, 1], 1))))
        l2_l3_3 = tf.reduce_mean(
            tf.maximum(
                0.0, tf.subtract(l3[:, 4], tf.expand_dims(l2[:, 2], 1))))
        hier_viol = tf.multiply(
            hier_vio_coef, (l1_l2 + l2_l3_1 + l2_l3_2 + l2_l3_3))
        hier_viol = tf.clip_by_value(hier_viol, epsilon, 5.)
        return hier_viol

    def scheduler(self, epoch):
        if epoch < 3:
            return 0.00005
        elif epoch < 5:
            return 0.00001
        else:
            return 0.000001


class HMLC_M(HMLC):

    def __init__(self, *args, **kwargs):
        super(HMLC_M, self).__init__(*args, **kwargs)
        self.global_dense4 = Dense(int(self.fp_len), activation="elu")
        self.global_dense5 = Dense(self.l3_len, activation=None)
        self.local_dense1_1 = Dense(self.fp_len, activation="elu")
        self.local_dense2_1 = Dense(self.fp_len, activation="elu")
        self.local_dense3_1 = Dense(self.fp_len, activation="elu")

    def call(self, inputs, training=None):
        out_g1 = self.global_dense1(inputs)
        out_l1_1 = self.local_dense1_1(out_g1)
        out_l1 = self.local_dense1(out_l1_1)

        in_g2 = self.add1([inputs, out_g1])
        out_g2 = self.global_dense2(in_g2)
        out_l2_1 = self.local_dense2_1(out_g2)
        out_l2 = self.local_dense2(out_l2_1)

        in_g3 = self.add2([inputs, out_g2])
        out_g3 = self.global_dense3(in_g3)
        out_l3_1 = self.local_dense3_1(out_g3)
        out_l3 = self.local_dense3(out_l3_1)

        in_g4 = self.add3([inputs, out_g3])
        out_g4 = self.global_dense4(in_g4)

        in_g5 = self.dropout(out_g4)
        out_g5 = self.global_dense5(in_g5)

        out_global = tfm.multiply(self.beta, out_g5)
        out_local = tfm.multiply(1-self.beta, out_l3)

        global_local_sum = self.add4([out_global, out_local])

        final_out = tf.sigmoid(tf.concat(
            [out_l1, out_l2, out_l3, global_local_sum], axis=1))
        return final_out

    def scheduler(self, epoch):
        if epoch < 3:
            return 0.00001
        elif epoch < 5:
            return 0.000005
        else:
            return 0.000001


class HMLC_L(HMLC_M):

    def __init__(self, *args, **kwargs):
        super(HMLC_L, self).__init__(*args, **kwargs)
        self.global_dense5 = Dense(int(self.fp_len), activation="elu")
        self.global_dense6 = Dense(self.l3_len, activation=None)
        self.local_dense1_2 = Dense(int(self.fp_len), activation="elu")
        self.local_dense2_2 = Dense(int(self.fp_len), activation="elu")
        self.local_dense3_2 = Dense(int(self.fp_len), activation="elu")

    def call(self, inputs, training=None):
        out_g1 = self.global_dense1(inputs)
        out_l1_1 = self.local_dense1_1(out_g1)
        out_l1_2 = self.local_dense1_2(out_l1_1)
        out_l1 = self.local_dense1(out_l1_2)

        in_g2 = self.add1([inputs, out_g1])
        out_g2 = self.global_dense2(in_g2)
        out_l2_1 = self.local_dense2_1(out_g2)
        out_l2_2 = self.local_dense2_2(out_l2_1)
        out_l2 = self.local_dense2(out_l2_2)

        in_g3 = self.add2([inputs, out_g2])
        out_g3 = self.global_dense3(in_g3)
        out_l3_1 = self.local_dense3_1(out_g3)
        out_l3_2 = self.local_dense3_2(out_l3_1)
        out_l3 = self.local_dense3(out_l3_2)

        in_g4 = self.add3([inputs, out_g3])
        out_g4 = self.global_dense4(in_g4)

        in_g5 = self.dropout(out_g4)
        out_g5 = self.global_dense5(in_g5)

        in_g6 = self.dropout(out_g5)
        out_g6 = self.global_dense6(in_g6)

        out_global = tfm.multiply(self.beta, out_g6)
        out_local = tfm.multiply(1-self.beta, out_l3)

        global_local_sum = self.add4([out_global, out_local])

        final_out = tf.sigmoid(tf.concat(
            [out_l1, out_l2, out_l3, global_local_sum], axis=1))
        return final_out

    def scheduler(self, epoch):
        if epoch < 3:
            return 0.00001
        elif epoch < 5:
            return 0.000001
        else:
            return 0.0000001


class HMLC_XL(HMLC_L):

    def __init__(self, *args, **kwargs):
        super(HMLC_XL, self).__init__(*args, **kwargs)
        self.global_dense1_2 = Dense(self.fp_len, activation="elu")
        self.global_dense2_2 = Dense(self.fp_len, activation="elu")
        self.global_dense3_2 = Dense(self.fp_len, activation="elu")

    def call(self, inputs, training=None):
        out_g1 = self.global_dense1(inputs)
        out_g1 = self.global_dense1_2(out_g1)
        out_l1_1 = self.local_dense1_1(out_g1)
        out_l1_2 = self.local_dense1_2(out_l1_1)
        out_l1 = self.local_dense1(out_l1_2)

        in_g2 = self.add1([inputs, out_g1])
        out_g2 = self.global_dense2(in_g2)
        out_g2 = self.global_dense2_2(out_g2)
        out_l2_1 = self.local_dense2_1(out_g2)
        out_l2_2 = self.local_dense2_2(out_l2_1)
        out_l2 = self.local_dense2(out_l2_2)

        in_g3 = self.add2([inputs, out_g2])
        out_g3 = self.global_dense3(in_g3)
        out_g3 = self.global_dense3_2(out_g3)
        out_l3_1 = self.local_dense3_1(out_g3)
        out_l3_2 = self.local_dense3_2(out_l3_1)
        out_l3 = self.local_dense3(out_l3_2)

        in_g4 = self.add3([inputs, out_g3])
        out_g4 = self.global_dense4(in_g4)

        in_g5 = self.dropout(out_g4)
        out_g5 = self.global_dense5(in_g5)

        in_g6 = self.dropout(out_g5)
        out_g6 = self.global_dense6(in_g6)

        out_global = tfm.multiply(self.beta, out_g6)
        out_local = tfm.multiply(1-self.beta, out_l3)

        global_local_sum = self.add4([out_global, out_local])

        final_out = tf.sigmoid(tf.concat(
            [out_l1, out_l2, out_l3, global_local_sum], axis=1))
        return final_out

    def scheduler(self, epoch):
        if epoch < 3:
            return 0.000005
        elif epoch < 5:
            return 0.000001
        else:
            return 0.0000001


class HMLC_XXL(HMLC_XL):

    def __init__(self, *args, **kwargs):
        super(HMLC_XXL, self).__init__(*args, **kwargs)
        self.global_dense1_3 = Dense(self.fp_len, activation="elu")
        self.global_dense2_3 = Dense(self.fp_len, activation="elu")
        self.global_dense3_3 = Dense(self.fp_len, activation="elu")

    def call(self, inputs, training=None):
        out_g1 = self.global_dense1(inputs)
        out_g1 = self.global_dense1_2(out_g1)
        out_g1 = self.global_dense1_3(out_g1)
        out_l1_1 = self.local_dense1_1(out_g1)
        out_l1_2 = self.local_dense1_2(out_l1_1)
        out_l1 = self.local_dense1(out_l1_2)

        in_g2 = self.add1([inputs, out_g1])
        out_g2 = self.global_dense2(in_g2)
        out_g2 = self.global_dense2_2(out_g2)
        out_g2 = self.global_dense2_3(out_g2)
        out_l2_1 = self.local_dense2_1(in_g2)
        out_l2_2 = self.local_dense2_2(out_l2_1)
        out_l2 = self.local_dense2(out_l2_2)

        in_g3 = self.add2([inputs, out_g2])
        out_g3 = self.global_dense3(in_g3)
        out_g3 = self.global_dense3_2(out_g3)
        out_g3 = self.global_dense3_3(out_g3)
        out_l3_1 = self.local_dense3_1(out_g3)
        out_l3_2 = self.local_dense3_2(out_l3_1)
        out_l3 = self.local_dense3(out_l3_2)

        in_g4 = self.add3([inputs, out_g3])
        out_g4 = self.global_dense4(in_g4)

        in_g5 = self.dropout(out_g4)
        out_g5 = self.global_dense5(in_g5)

        in_g6 = self.dropout(out_g5)
        out_g6 = self.global_dense6(in_g6)

        out_global = tfm.multiply(self.beta, out_g6)
        out_local = tfm.multiply(1-self.beta, out_l3)

        global_local_sum = self.add4([out_global, out_local])

        final_out = tf.sigmoid(tf.concat(
            [out_l1, out_l2, out_l3, global_local_sum], axis=1))
        return final_out

    def scheduler(self, epoch):
        if epoch < 3:
            return 0.000005
        elif epoch < 5:
            return 0.000001
        else:
            return 0.0000001
