from sklearn.metrics import accuracy_score,f1_score, classification_report,mean_absolute_error,precision_score,recall_score
import src.Evaluation.Plotting as Plotting
import numpy as np
from contextlib import redirect_stdout
def Results(model, X_test, Y_test, modelName, history=None,epochs=None):
    Y_predict=model.predict(X_test)
    res_file = open('results/'+modelName+ '/results.txt', 'w')
    res_file.write('----------MODEL DESCRIPTION----------')
    res_file.write('\nModel name: '+str(modelName))
    if(modelName!= "RandomForestClassifier"):
        Plotting.Prot(history, X_test, Y_test, modelName)    
        pred_file = open('results/'+ modelName + '/probabilities.txt', 'w')
        pred_file.write('----------PROBABILITY----------\n')
        print(type(Y_predict))
        np.savetxt(pred_file, Y_predict)  
        pred_file.close()
        Y_predict = np.argmax(Y_predict, axis=1)
        Score = model.evaluate(X_test, Y_test, verbose=1)
        n_layers=len(model.layers)
        print("Number of layers: "+str(n_layers))
        hist = history.history['val_accuracy']
        epochs = np.argmax(hist)
        print("The best number of ephochs: "+str(epochs))
        res_file.write('\nThe best number of epochs: '+str(epochs))
        res_file.write('\nNumber of layers: '+str(n_layers))
        print("Loss: "+ str(Score[0]))
    
    Accuracy =str(accuracy_score(Y_test,Y_predict))
    MSE=str(mean_absolute_error(Y_test,Y_predict))
    Precision=str(precision_score(Y_test,Y_predict, average = 'macro'))
    Recall = str(recall_score(Y_test,Y_predict, average = 'macro'))
    F1_score=str(f1_score(Y_test,Y_predict, average = 'macro'))
    Clf_report=str(classification_report(Y_test,Y_predict))

    print("Accuracy: "+ Accuracy)
    print("Mean Absolute Error: "+MSE)
    print("Precision: " +'\n'+Precision)
    print("Recall: " +'\n'+Recall)
    print("F1_score: " +'\n'+ F1_score)
    print("Classification Report: " +'\n'+ Clf_report)
    
#     with redirect_stdout(res_file):
#         model.summary()
    res_file.write('\n\n---------------RESULTS---------------')
    if(modelName!= "RandomForestClassifier"):
        res_file.write('\nThe test loss: ' + str(Score[0]))
    res_file.write('\nThe test accuracy: ' + Accuracy+
                '\nThe test MSE: ' + MSE + 
                '\nThe test precision: ' + Precision+
                '\nThe test recall: ' + Recall+
                '\nThe test F1_score: ' + F1_score+
                '\nClassification Report:\n' + Clf_report)
    res_file.close()
    pred_f = open('results/'+modelName+  '/predictions.txt', 'w')
    pred_f.write('----------PREDICTION----------\n')
    pred_f.write(str(Y_predict))
    pred_f.close()

    
