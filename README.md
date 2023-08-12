# Disaster Response Pipeline Project

## Installation

The code uses Python scripts. There are also some Jupyter notebook files which were used for the preparation and testing of the final code. 

## Motivation

The purpose of this project is to analyse over 20000 messages that were sent during natural disasters either via social media or directly to natural disaster response organisation and allocate them to 36 different categories (e.g. aid related, request, medical help e.t.c.).  

## File description

1. process_data.py: The python script that reads the initial disaster and category messages, merges the two datasets and prepares a final clean dataset that will be analysed by next file.

2. train_classifier.py: The python script that splits the data into train and test dataset and then applies several classification Machine Learning algorithms in order to find the most accurate one.

3. run.py: The python script that reads the clean dataset from file 1 and the optimum model from file 2 and then creates 3 descriptive graphs and an app using Flask that takes a message as an input and classifies it across the 36 categories  

4. Some Jupyter notebook files that were used in preparation of the Python scripts.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
