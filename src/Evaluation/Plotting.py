import matplotlib.pyplot as plt
import os


def plotting(history,  model_name, path="results"):
    if not os.path.isdir(path+"/"+model_name+"/plots"):
        os.mkdir(path+"/"+model_name+"/plots")
    plt.figure(1)
    plt.grid(True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(path+'/' + model_name + '/plots/accuracy.png')
    plt.figure(2)
    plt.grid(True)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig(path+'/' + model_name + '/plots/loss.png')
