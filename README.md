# Disaster-Response-Project-Udacity
Python project to classify disaster response messages using Machine Learning models and create a web app with Flask.

## Contents

1. [Installation](#Installation)
2. [Project Description](#Project-Description)
3. [File Description](#File-Description)
4. [To Execute Pipelines](#To-Execute-Pipelines)
6. [Results](#Results)
7. [Acknowledgments](#Acknowledgements)


## Installation
The necessary libraries to run the code :
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
This project is a small attempt to classify messages that can inturn increase efficiency of disaster response systems. Dataset for the project which consists of 30,000 messages was provided by [Figure eight](https://appen.com/). The messages in this dataset is categorised into 36 different categories like, 'medical_products','aid_centres','requests' etc.
Machine learning model (Linear SVC) is used to classify the messages into these 36 categories. A webpage is created using the dataset and model where user input messages (eg: Need clothing , Shortage in water, Need Medical supplies etc) can be classified.

## File Description

```
- app               # flask app files
| - template        # file with html code and python code for the webpage
| |- master.html    # main page of web app
| |- go.html        # classification result page of web app
|- run.py           # Flask file that runs app

- data
|- disaster_categories.csv    # data with categories of each messsage
|- disaster_messages.csv      # data with messages to classify
|- process_data.py            # code to clean and load data to sql database
|- DisasterResponse.db        # database to save clean data to

- models
|- train_classifier.py       # code with machine learning model code and evaluation matrix
|- classifier.pkl            # saved model 

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
     
## To Execute Pipelines

Execute the following code in the terminal to run the ETL_pipeline (process_data.py) and Machine_learning pipeline (train_classifier.py) respectively

 * `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
 * `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
 
 To run the web app (run.py) execute the following code
 
 * `python run.py`
 
 To open the web app
 
 * ```env|grep WORK                   # to create a virtual environment ```
 * Get the SPACEID and SPACEDOMAIN 
 * Type the following in new web browser __https://SPACEID-3001.SPACEDOMAIN__  
 
 
## Results

A web app was created using the dataset, where a user input message related to a disaster can be classified into one of the 36 categories provided.
  * Web_app Visualisations
  
![Web_app_visualisations](https://github.com/athira-g-96/Disaster-Response-Project-Udacity/blob/main/web_pics/visualisations.png)
  * Web_app Classification

![Web_app](https://github.com/athira-g-96/Disaster-Response-Project-Udacity/blob/main/web_pics/webapp.png)

## Acknowledgments

Credits to Figure Eight for providing the dataset 

## Note

The dataset is imbalanced and Linear SVC model was used to classify the messages as the model runs faster compared to other models tried (SGDClassifier,Random Forest, KNN) therefore the accuracy of the model is low.






