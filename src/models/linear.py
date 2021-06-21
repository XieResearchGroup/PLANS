from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout


class Linear_S(Model):
    def __init__(self, fp_len=2048, drop_rate=0.3, out_len=32, activation="softmax"):
        super(Linear_S, self).__init__()
        self.fp_len = fp_len
        self.drop_rate = drop_rate
        self.out_len = out_len
        self.dense1 = Dense(fp_len * 2, activation="relu")
        self.dense2 = Dense(fp_len * 4, activation="relu")
        self.dense3 = Dense(fp_len * 4, activation="relu")
        self.dense4 = Dense(fp_len * 2, activation="relu")
        self.dense5 = Dense(fp_len, activation="relu")
        self.dense6 = Dense(int(fp_len // 2), activation="relu")
        self.dense7 = Dense(int(fp_len // 4), activation="relu")
        self.dropout = Dropout(drop_rate)

        self.dense_out = Dense(out_len, activation=activation)

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

    def scheduler(self, epoch, init_rate=1e-6):
        if epoch < 20:
            return init_rate
        elif epoch < 40:
            return init_rate / 10
        else:
            return init_rate / 100

    def __str__(self):
        return "Linear_S"

    @property
    def name(self):
        return self.__str__()


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
        self.dense2_2 = Dense(self.fp_len * 6, activation="relu")
        self.dense2_3 = Dense(self.fp_len * 6, activation="relu")

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense1_2(x)
        x = self.dense2(x)
        x = self.dense2_2(x)
        x = self.dense2_3(x)
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
        return "Linear_L"
