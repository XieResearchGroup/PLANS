import matplotlib.pyplot as plt
import random


data = list()
for i in range(10):
    his = dict()
    his["loss"] = [random.randrange(0, 250)/100 for _ in range(20)]
    his["acc"] = [random.randrange(0, 100)/100 for _ in range(20)]
    his["val_loss"] = [random.randrange(0, 250)/100 for _ in range(20)]
    his["val_acc"] = [random.randrange(0, 100)/100 for _ in range(20)]
    data.append(his)


def plot_history(train_his):
    r""" Plot the training history
    train_his: Tensorflow History callback object
    path: path to save the ploting
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[12.8, 4.8])
    x_max = 0
    y_loss_max = 0
    y_acc_max = 0
    for i, his in enumerate(train_his):
        train_loss = his["loss"]
        val_loss = his["val_loss"]
        train_acc = his["acc"]
        val_acc = his["val_acc"]
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
        current_handles, current_labels = ax.get_legend_handles_labels()
        current_handles = [
            h for _, h in sorted(zip(current_labels, current_handles))]
        current_labels.sort()
        lgd = ax.legend(
            current_handles, current_labels,
            bbox_to_anchor=(0., -0.1, 1, 0), loc=9,
            ncol=2, mode=None, borderaxespad=0.)
        lgds.append(lgd)

    axes[0].set(ylim=[0, y_loss_max+0.1], title="loss")
    axes[1].set(ylim=[0, y_acc_max+0.1], title="accuracy")

    fig.savefig(
        "training_results.png", bbox_extra_artists=lgd, bbox_inches='tight')


plot_history(data)
