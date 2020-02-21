from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout


class Linear_S(Model):

    def __init__(self, fp_len=2048, drop_rate=0.3, out_len=32):
        super(Linear_S, self).__init__()
        self.fp_len = fp_len
        self.dense1 = Dense(fp_len * 2, activation="relu")
        self.dense2 = Dense(fp_len * 4, activation="relu")
        self.dense3 = Dense(fp_len * 4, activation="relu")
        self.dense4 = Dense(fp_len * 2, activation="relu")
        self.dense5 = Dense(fp_len, activation="relu")
        self.dense6 = Dense(int(fp_len//2), activation="relu")
        self.dense7 = Dense(int(fp_len//4), activation="relu")
        self.dropout = Dropout(drop_rate)
        self.dense_out = Dense(out_len, activation="softmax")

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dropout(x)
        x = self.dense_out(x)
        return x

    def scheduler(self, epoch):
        if epoch < 20:
            return 0.00001
        elif epoch < 40:
            return 1e-6
        else:
            return 1e-7

    def __str__(self):
        return "Linear_S"


class Linear_M(Linear_S):

    def __init__(self, *args, **kwargs):
        super(Linear_M, self).__init__(*args, **kwargs)
        self.dense1_2 = Dense(self.fp_len * 3, activation="relu")
        self.dense3_2 = Dense(self.fp_len * 3, activation="relu")

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense1_2(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense3_2(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dropout(x)
        x = self.dense_out(x)
        return x

    def __str__(self):
        return "Linear_M"


class Linear_L(Linear_M):

    def __init__(self, *args, **kwargs):
        super(Linear_L, self).__init__(*args, **kwargs)
        self.dense4_2 = Dense(self.fp_len * 6, activation="relu")
        self.dense4_3 = Dense(self.fp_len * 6, activation="relu")

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense1_2(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense3_2(x)
        x = self.dense4(x)
        x = self.dense4_2(x)
        x = self.dense4_3(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dropout(x)
        x = self.dense_out(x)
        return x

    def __str__(self):
        return "Linear_L"
