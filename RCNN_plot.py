import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt


def plot_training_results(losses, weights):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, 'o-', color='blue')
    plt.title("Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.hist(weights[-1].numpy(), bins=50, color='blue', edgecolor='blue', alpha=0.3)
    plt.title("Weight Distribution After Training")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()