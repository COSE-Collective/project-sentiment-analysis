from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_absolute_error, precision_score, \
    recall_score
import src.Evaluation.Plotting as Plotting
import numpy as np


def model_evaluation(model, x_test, y_test, model_name, history=None):
    Plotting.plotting(history, model_name)

    Y_prob = model.predict(x_test)
    prob_file = open('results/' + model_name + '/probabilities.txt', 'w')
    prob_file.write('----------PROBABILITY----------\n')
    np.savetxt(prob_file, Y_prob)
    prob_file.close()

    Y_predict = np.argmax(Y_prob, axis=1)
    pred_f = open('results/' + model_name + '/predictions.txt', 'w')
    pred_f.write('----------PREDICTION----------\n')
    np.savetxt(pred_f, Y_predict)
    pred_f.close()

    Accuracy = str(accuracy_score(y_test, Y_predict))
    MSE = str(mean_absolute_error(y_test, Y_predict))
    Precision = str(precision_score(y_test, Y_predict, average='macro'))
    Recall = str(recall_score(y_test, Y_predict, average='macro'))
    F1_score = str(f1_score(y_test, Y_predict, average='macro'))
    Clf_report = str(classification_report(y_test, Y_predict, target_names=['Negative', 'Neutral', 'Positive']))
    Score = model.evaluate(x_test, y_test, verbose=1)
    n_layers = len(model.layers)
    hist = history.history['val_accuracy']
    epochs = np.argmax(hist)

    print('----------MODEL DESCRIPTION----------')
    print('Model name: ', str(model_name))
    print('The best number of epochs: ', str(epochs))
    print('Number of layers: ', str(n_layers))
    print("Loss: ", str(Score[0]))
    print("Accuracy:", Accuracy)
    print("Mean Absolute Error:", MSE)
    print("Precision:", Precision)
    print("Recall:", Recall)
    print("F1_score:", F1_score)
    print("Classification Report: " + '\n' + Clf_report)
    res_file = open('results/' + model_name + '/results.txt', 'w')
    res_file.write('----------MODEL DESCRIPTION----------')
    res_file.write('\nModel name: ' + str(model_name))
    res_file.write('\nThe best number of epochs: ' + str(epochs))
    res_file.write('\nNumber of layers: ' + str(n_layers))

    res_file.write('\n\n---------------RESULTS---------------')
    res_file.write('\nThe test loss: ' + str(Score[0]))
    res_file.write('\nThe test accuracy: ' + Accuracy +
                   '\nThe test MSE: ' + MSE +
                   '\nThe test precision: ' + Precision +
                   '\nThe test recall: ' + Recall +
                   '\nThe test F1_score: ' + F1_score +
                   '\nClassification Report:\n' + Clf_report)
    res_file.close()
