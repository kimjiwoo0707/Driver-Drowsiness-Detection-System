import matplotlib.pyplot as plt

def plot_training_history(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, "b", label="Training Accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation Accuracy")
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, "b", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.legend()
    plt.show()
