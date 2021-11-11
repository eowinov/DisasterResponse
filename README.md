# Disaster Response Pipeline Project

## Project Overview
This project uses disaster data from Figure Eight containing messages and categories for the messages. Each message can be assigned to categories (i.e. 'medical_help','water','shelter',..). The aim of this project is to create a ML Pipeline that uses NLP techniques to classify the text messages to the right category and showing results via webapp. The webapp shows some data visuals on the one hand. On the other hand it is possible to predict the message categories for any desired messages.

## Installation
The project is based on Python 3.0 (Anaconda distribution. The following libraries are used:
1. numpy
2. pandas
3. sqlalchemy
4. sqlite3
5. nltk
6. sklearn
7. pickle


## Data 
The data used in this project is is from Figue Eight. The dataset consists of 2 csv.-files:
- disaster_messages.csv
- disaster_categories.csv

## Files in Repository
The folder "data" contains the csv data files, the python file process_data.py that extracts, transforms and loads the data into a SQL database. After creation the database is also stored in the data folder, because it was used as filepath when calling the process_data.py script.

In the folder "models", there is a python script train_classifier.py that extracts data from the created database, builds a ML pipeline, trains a model and stores the model in the classifier.pkl file.

The folder "app" holds the html files used for the flask webapp. The run.py file starts the app.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (instead of 0.0.0.0 I have to use my local IP)

## Webapp Results
![](/main_page_visuals.jgp "Main Page Visuals")
![](/message_classification.jpg "Message Classification")

## Acknowledgements
1. [Figure Eight](https://appen.com/) for providing the data set for this project
2. [Udacity](https://www.udacity.com/) for providing this interesting NLP project

