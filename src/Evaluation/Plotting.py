import matplotlib.pyplot as plt


def plotting(history,  model_name):
    plt.figure(1)
    plt.grid(True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('results/' + model_name + '/plots/accuracy.png')
    plt.figure(2)
    plt.grid(True)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig('results/' + model_name + '/plots/loss.png')
