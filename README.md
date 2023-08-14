# Disaster Response Pipeline Project

## Installation

The code uses Python scripts. There are also some Jupyter notebook files which were used for the preparation and testing of the final code. 

## Motivation

This project is part of my Data Science Nanodegree training via Udacity. The purpose of this project is to analyse over 20000 messages that were sent during natural disasters either via social media or directly to natural disaster response organisation and allocate them to 36 different categories (e.g. aid related, request, medical help e.t.c.).  

## File description
- Folder data: 
    1. process_data.py: The python script that reads the initial disaster and category messages, merges the two datasets and 
    prepares a final clean dataset that will be analysed by the models in train_classifier.py file.

    2. disaster_categories.csv: The csv file provided by Udacity containing the categories in which each message is 
       allocated

    3. disaster_messages.csv: The csv file provided by Udacity containing the initial messages sent to disaster response 
       organisations or posted via social media

    4. DisasterResponse.db: The database in which the clean dataset from the first file (process_data.py) will be stored

- Folder models:
     1. train_classifier.py: The python script that splits the data into train and test dataset and then applies several 
        classification Machine Learning algorithms in order to find the most accurate one (final model).

     2. Disaster_model.pkl: The final model created by train_classifier.py. It was stored in this folder

 - Folder app:
      1. run.py: The python script that reads the clean dataset and the optimum model (Disaster_model.pkl above) and then 
         creates 3 descriptive graphs and an app using Flask that takes a message as an input and classifies it across the 
         36 categories
      2. Folder templates:
           1. master.html : The html code provided by Udacity that creates the three graphs in the app
           2. go.html: The html code provided by Udacity that accepts a message as an input in the app and classifies it 
              across the 36 categories 
 
 - Some Jupyter notebook files that were used in preparation of the Python scripts.


### Licensing, Authors, Acknowledgements
Files disaster_categories.csv and disaster_messages.csv where provided by Udacity (https://www.udacity.com/) for the purposes of Data Science Nanodegree training. Udacity also provided the code for master.html and go.html and a template for the three python files (run.py,process_data.py,train_classifier.py) as well the Jupyter Notebooks.   

### Instructions:
1. Copy the files from the three folders (app,data,models) in same named folders in a python workspace (project's root 
   directory)                       
                  
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run the web app: `python run.py`

5. Click the `PREVIEW` button to open the homepage
