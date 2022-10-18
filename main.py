import argparse
from src.Models.RandomForestClassifier.RandomForestClassifier import RandomForest
from src.Models.LSTM.LSTM import LSTM
from src.Models.BiLSTM.BiLSTM import BiLSTM
import src.Loading.DataLoading as DataLoading
import src.Evaluation.ResultsEvaluation as ResultsEvaluation
def parse_args():
    """
    Parses the bash introduced arguments.
    """

    parser = argparse.ArgumentParser(description = "Sentiment analysis smart bash runner.")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('--model_name', nargs = '?',required = True,
                          help = 'Especify the model name: RF, LSTM, BiLSTM')
    return parser.parse_args()


def main(args):
    X_train, X_test, Y_train, Y_test, dataset= DataLoading.LoadDataset()
    data = [X_train, X_test, Y_train, Y_test]

    if args.model_name == 'RF':
        model = RandomForest(dataset)
        X_test, Y_test = model.Train()
        loaded_model = DataLoading.LoadModel(args.model_name)
        print('Data estimation')
        ResultsEvaluation.Results(loaded_model, X_test, Y_test, "RandomForestClassifier")
        print('Completed')
        return
    elif args.model_name == 'LSTM':
        model = LSTM(data)
    elif args.model_name == 'BiLSTM':
        model = BiLSTM(data)
    else:
        print('Error: Model does not exist')
        return
    history, X_test, epochs= model.Train()
    loaded_model = DataLoading.LoadModel(args.model_name)
    print('Results estimation')
    ResultsEvaluation.Results(loaded_model, X_test, Y_test, args.model_name, history,epochs)
    print('Completed')
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
    