# Disaster-Response-Project-Udacity
Python project to classify disaster response messages using Machine Learning models and to create a web app using the dataset and model with Flask.

## Contents

1. [Installation](#Installation)
2. [Project Description](#Project-Description)
3. [File Description](#File-Description)
4. [Results](#Results)
5. [Acknowledgments](#Acknowledgements)


## Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.

Libraries used are:
* Pandas
* Numy
* Scikit learn
* sqlalchemy
* plotly
* flask
* json
* nltk
* itertools
* collections
* pickle

## Project Description
This project is a small attempt to classify messages related to disaster so as to increase the efficiency of disaster response. Dataset of approximately 30,000 messages was provided by [Figure eight](https://appen.com/). The messages in the dataset is categorised into 36 different categories like, 'medical_products','aid_centres','requests' etc.
Machine learning model ,Linear SVC is used to classify the messages into these 36 categories. A webpage is created using the dataset and model where any user input messages related to a disaster (eg: Need clothing , Shortage in water, Need Medical supplies etc) can be classified.

## File Description

```- app # flask app files
| - template # file with html code and python code for the webpage
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data with categories of each messsage
|- disaster_messages.csv  # data with messages to classify
|- process_data.py # code to clean and load data to sql database
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # code with machine learning model code and evaluation matrix
|- classifier.pkl  # saved model 

- README.md
 ```
 ### `.py` files :
 
 * `process_data.py` : This file contains code to clean data and save it to 
     * Loads both csv files (disaster_messages.csv, disaster_categories.csv )
     * Merges the two datasets to get a dataframe
     * Drops duplicates and Cleans the data
     * Stores it in a SQLite database( DisasterResponse.db)

* ` train_classifier.py` : File contains code to select best ML model and evaluation of the model 
     * Gets data from sql database
     * Splits the data into independent and dependent variables:
           X (independent value) : 'message'
           Y (dependent value) : array of all 36 categories
     * Splits X and Y into training and testing data
     * Creates a pipeline with TFIDF as feature and Linear SVC as ML model
     * Fits the model
     * Runs GridSearchCV to get best parameters for the model
     * Predicts for the test set
     * Evaluates the model using classification report and accuracy score
     * Saves the model as a pickle file (classifier.pkl)

* `run.py` : File contains
     *  code to classify the user input message in web app
     *  code for visualisations of the training data of the model

## Results

A web app was created using the dataset, where a user input message related to a disaster can be classified into one of the 36 categories provided.
![Web_app_visualisations]()
![Web_app_visualisations]()
![Web_app]()

## Acknowledgments

Credits to Figure Eight for providing the dataset 

## Note

The dataset is imbalanced and Linear SVC model was used to classify the messages as the model runs faster compared to other models tried (SGDClassifier,Random Forest, KNN) therefore the accuracy of the model is low.






