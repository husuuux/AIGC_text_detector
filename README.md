# Logistic Regression and Feature Extraction-based Model for Detection of AI-Generated Texts

Suxin Hu, Guoliang Wang, and Jinghong Li

The codes of our paper "Logistic Regression and Feature Extraction-based Model for Detection of AI-Generated Texts", prepared for Mathematics Modeling Competition of SWU 2024.

## Detector Models

We have open-sourced detector models in the paper as follows.

## About the Dataset

Here we provide the dataset, which contains around 400 abstracts from essays of MIT. We also provide the original documents and their chat4.0-variants for your ease of use. 

#### Data Preprocessing

Here we provide a divided version of the abstract of MIT essays. In this version, all answers are cleaned. However, please use the original version of essays for all experiments in our paper, as we have used the variants of essays in the model training.

##  Preparation

- Install requirement packages:

```shell
pip install -r requirements.txt
```

- Download nltk package punct (This step could be done by ```nltk``` api: ```nltk.download('punkt')```)

- Download pretrained models (This step could be automatically done by ```transformers```)

- Prepare a certain model to compare with, and put it in the folder ```models```, so that you can test the accuracy of the model.

- Change the directory of the nltk_data in the utils.py file.

Before running, the directory should contain the following files:

- The file ```init.py``` to initialize as a package and folder ```runs``` for storing the trained models are empty.

```
├── data
│   ├── chat4.0 Economicspilot
│   │   ├── text1.txt
│   │   ├── text2.txt
│   │   ├── ……
│   │   └── text126.txt
│   ├── Chat4.0 MIT physics pilot
│   │   ├── text1.txt
│   │   ├── text2.txt
│   │   ├── ……
│   │   └── text100.txt
│   ├── MIT physics pilot
│   │   ├── text1.txt
│   │   ├── text2.txt
│   │   ├── ……
│   │   └── text103.txt
│   ├── MIT_Dept. of Economicspilot
│   │   ├── text1.txt
│   │   ├── text2.txt
│   │   ├── ……
│   │   └── text103.txt
│   ├── demo1.txt
│   ├── demo2.txt
│   └── Data interpretation
├── models
│   ├── trained_model
│   │   ├── logistic_regression_model.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── runs
├── results
│   ├── Othermodel_test_result.png
│   ├── prediction_results.csv
│   ├── train_result.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── important_features.png
│   └── test_result.png
├── README.md
├── utils.py
├── init.py
├── test.py
├── othermodel_test.py
├── requirements.txt
├── othermodel_test_algorithm.pdf
├── model_training.pdf
├── model_feature_extraction.pdf
├── train.py
└── demo.py
```

## Training

The script for training is ```train.py```.



