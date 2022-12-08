import argparse
from src.Models.LSTM.LSTM import LSTM
from src.Models.BiLSTM.BiLSTM import BiLSTM
import src.Loading.DataLoading as DataLoading
import src.Evaluation.ResultsEvaluation as ResultsEvaluation
from src.Models.BERT.BERT import BERT
from src.Models.RoBERTa.RoBERTa import ROBERTA

def parse_args():
    """
    Parses the bash introduced arguments.
    """

    parser = argparse.ArgumentParser(description="Sentiment analysis smart bash runner.")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('--model_name', nargs='?', required=True,
                          help='Specify the model name: LSTM, BiLSTM, BERT, RoBERTa')
    return parser.parse_args()


def main(args):
    X_train, X_test, Y_train, Y_test, dataset = DataLoading.dataset_loading()
    data = [X_train, X_test, Y_train, Y_test]

    if args.model_name == 'LSTM':
        model = LSTM(data)
    elif args.model_name == 'BiLSTM':
        model = BiLSTM(data)
    elif args.model_name == 'BERT':
        model = BERT(data)
    elif args.model_name == 'RoBERTa':
        model = ROBERTA(data)
    else:
        print('Error: Model does not exist')
        return
    model_loaded, history, X_test = model.Train()
    print('Results estimation')
    ResultsEvaluation.model_evaluation(model_loaded, X_test, Y_test, args.model_name, history)
    print('Completed')


if __name__ == "__main__":
    args = parse_args()
    main(args)
