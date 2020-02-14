import numpy as np

from ..utils.mixup import mixup

batch_size = 5
rep = 2
shuffle = True

a = np.round(np.random.rand(batch_size, 4))
label_a = np.round(np.random.rand(batch_size, 5))
b = np.round(np.random.rand(batch_size, 4))
label_b = np.round(np.random.rand(batch_size, 5))

sample, label = mixup(0.4, 0.4, a, label_a,
                      sample_2=b, label_2=label_b, repeat=rep, shuffle=shuffle)
print("a is: \n{}".format(a))
print("b is: \n{}".format(b))
print("mixed samples are: \n{}".format(sample))
print()
print("label of a is: \n{}".format(label_a))
print("label of b is: \n{}".format(label_b))
print("mixed labels are: \n{}".format(label))
print()

sample, label = mixup(0.4, 0.4, a, label_a, repeat=rep, shuffle=shuffle)
print("mix the samples in the same group by random permutation.")
print("result: \n{}".format(sample))
print("labels: \n{}".format(label))
