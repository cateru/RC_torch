import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_training_results(losses, weights):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, 'o-', color='blue')
    plt.title("Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.hist(weights[-1].numpy().flatten(), bins=50, color='blue', edgecolor='blue', alpha=0.3)
    plt.title("Weight Distribution After Training")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_readout_weights(self, show_bias=True):
        W = self.readout.weight.detach().cpu()
        plt.figure(figsize=(10, 5))
        plt.imshow(W, aspect='auto')
        plt.colorbar()
        plt.title("Readout weights (W matrix)")
        plt.xlabel("Reservoir features")
        plt.ylabel("Output classes")
        plt.show()
        if show_bias:
            b = self.readout.bias.detach().cpu()
            plt.figure(figsize=(6, 3))
            plt.bar(range(len(b)), b.numpy())
            plt.title("Readout bias")
            plt.xlabel("Class")
            plt.show()


def plot_accuracy(accuracies):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(accuracies)+1),
             accuracies,
             marker='o')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()