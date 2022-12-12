# Sentiment analysis project
This repository contains 4 models (LSTM, BiLSTM, pretrained BERT, pretrained RoBERTa) for sentiment analysis of tweets. 
> ### Project structure
> #### There are 3 folders and 1 file at this level. Let's talk about the purpose of each of those.
> - ./datasets -> Contains 10 file: 5 original datasets, 2 preprocessed training and 3 preprocessed testing datasets. The datasets were taken from Kaggle:
>   + https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis 
>   + https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=testdata.manual.2009.06.14.csv
> - ./results -> This folder contains the results of models evaluation and divided into 4 subfolders for each model. Each model has the following results: file with probability values(./probabilities.txt), file with prediction values(./predictions.txt), file with results(./results.txt), accuracy and loss  plots (./plots/accuracy.png, ./plots/loss.png)
> - ./src -> This folder contains following subfolders:
>   + ./DataPreprocessing-folder with the files for preprocessing the datasets
>   + ./Evaluation-folder with the files for results evaluation and drawing plots
>   + ./DataLoading-folder with the file for data and models loading
>   + ./Models-folder with the files for models building
> - main.py -> Is the execution script that will parse the commands you give from the terminal to your code.
To execute the main.py file the command must have the argument --model_name <model_name>, where <model_name> has 4 options: LSTM, BiLSTM, BERT, RoBERTa

Comment: The BERT and RoBERTa models are complex and have many parameters, so it is better to use Google Colab to train these models. 

> ### Environment setup
> #### Step 1
> Make sure you are on the directory level of this README file.
> #### Step 2
> Create a venv with python with the following command:\
> *python3 -m venv /path/to/new/virtual/environment*\
> In this case we can set the path to: *./venv*
> #### Step 3
> Install all the packages needed for the challenge with the following command:\
> *pip install -r requirements.txt*
> #### Step 4
> Activate the environment in your terminal:\
> *source ./venv/bin/activate*\
> You will see that (venv) has appeared in the beginning of your terminal line. With that you know the python environment in which you are operating in that terminal.