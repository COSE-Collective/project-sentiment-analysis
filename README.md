[![Python: 3839](https://img.shields.io/badge/python-3.8%20%7C%203.9-9cf)](https://docs.python.org/release/3.8.10/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blueviolet)](https://opensource.org/licenses/MIT)


# Project Sentiment Analysis

This repository contains 4 models (LSTM, BiLSTM, pre-trained BERT, pre-trained RoBERTa) for tweet/short post sentiment analysis.

## Project structure
### There are 2 folders and 1 file at this level. Let's talk about the purpose of each of those.
 - **./datasets** -> Contains 10 files: 5 original datasets, 2 pre-processed training and 3 pre-processed testing datasets. The datasets were taken from Kaggle:
   + https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis 
   + https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=testdata.manual.2009.06.14.csv
 - **./src** -> This folder contains the following subfolders:
   + **./DataPreprocessing** folder with the files for pre-processing the datasets.
   + **./Evaluation** folder with the files for results evaluation and drawing plots.
   + **./DataLoading** folder with the file for data and models loading.
   + **./Models** folder with the files that build the models.
 - ***main.py*** -> Is the execution script that will parse the commands you give from the terminal to your code.
To execute the main.py file the command must have two arguments:
   - --model_name <model_name>, where <model_name> has 4 options: LSTM, BiLSTM, BERT, RoBERTa
   - --results_folder <folder_name>

## Disclaimer
Because the implemented BERT and RoBERTa models are highly complex and have over 100M parameters, we recommend using tools like Google Colab in case a memory overload occurs. 

## License
This repository is under MIT License.

## References
- LSTM: https://ieeexplore.ieee.org/abstract/document/7778967
- BiLSTM: https://ieeexplore.ieee.org/abstract/document/8684825
- BERT: https://arxiv.org/abs/1810.04805
- RoBERTa: https://arxiv.org/abs/1907.11692


<p align="center">
  <img src="https://github.com/COSE-Collective/project-sentiment-analysis/blob/master/coselogo.png">
</p>
