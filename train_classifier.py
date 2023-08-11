# import all necessary libraries
import sys
import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report,precision_recall_fscore_support,f1_score
from sklearn.metrics import accuracy_score,make_scorer, precision_score, recall_score
import pickle
import time
import warnings

def load_data(database_filepath):
    """Reads the data from the sql table in sqlite database DisasterResponse.db

    Args:
    database_filepath: The path for the sqlite database
    
    Returns:
    X: The explanatory variable (messages) that will be used in the model
    Y: The 36 categories for which the model will predict values (0-1)
    Y.columns: The names of the 36 target variables
    """ 
    
    engine = create_engine('sqlite:///'+database_filepath)  # create a connection to the database 
    df = pd.read_sql_table('Disaster_category_messages',engine)  # read the data from the table in the db
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    return X,Y,Y.columns


def tokenize(text):
    """Tokenizer function to be used as an argument in the countvectorizer object later

    Args:
    text: A single string value
    
    Returns:
    All tokens in the text as a list
    """
    
    text=re.sub(r'[^0-9a-zA-Z]',' ',text.lower())  # Punctuation removed-Lowercase   
    tokens=word_tokenize(text)  # Split in words-put them in a list
    lemmatizer=WordNetLemmatizer()  # Create an object of WordNetLemmatizer class. Will be used in the loop below  
    
    # The tokens in the list will be lemmatized and added in a new list that will be returned
    clean_tokens=[]
    for tok in tokens:
        clean_tokens.append(lemmatizer.lemmatize(tok))  # The words of the previous list are lemmatized here
   
    return clean_tokens


def build_model():
    """This is a pipeline that creates a model that will train and fit data

    Args: None 
    
    Returns:
    The pipeline that has a fit and a predict method 
    """
        
    pipeline = Pipeline([
        # A CountVectorizer object is created -Tokenize function used
        ('vect', CountVectorizer(tokenizer=tokenize)),  
        
        # A TfidfTransformer object is used as a second step
        ('tfidf', TfidfTransformer()),    
        
        # RandomForestClassifier was chosen as the initial ML method for training and fitting
        ('clf', MultiOutputClassifier(RandomForestClassifier()))                
])
          
    return pipeline

def average_scoring_single_metric(Y_test, Y_pred,metric):
    
    """Creates a list of 36 scores (one for each target variable)
       depending on the metric that will be used

    Args:
    Y_test: The test dataset for target variables 
            after the train test split 
    Y_pred: The dataset of predicted values for target variables
            after fit and predict
    metric: The metric chosen from the list below:
            [f1_score,accuracy_score,recall_score,
            precision_score,classification_report]
    
    Returns:
    A list of 36 scores - one for each variable
    """
    
    target_vars=Y_test.columns  # A list with target variable names

    # For each of the target variables a score will be calculated
    # and added to a list
    
    scores = []
    for i in range(len(target_vars)):
        target_var = target_vars[i]
        Y_test_single_var=Y_test[target_var]  # Isolate the true values for
                                                   # the variable
            
        Y_pred_single_var = pd.Series(Y_pred[:,i])  #Isolate the predicted values
                                                    # for the variable
        
        # Calculate the score for the variable depending on the metric
        # The relevant function for each metric was imported at the beginning
        if metric=='f1_score':
            score = f1_score(Y_test_single_var, Y_pred_single_var,average='weighted')
        elif metric=='accuracy_score':
            score = accuracy_score(Y_test_single_var, Y_pred_single_var)
        elif metric=='recall_score':
            score = recall_score(Y_test_single_var, Y_pred_single_var,average='weighted')
        elif metric=='precision_score':
            score = precision_score(Y_test_single_var, Y_pred_single_var,average='weighted')
        elif metric=='classification_report': 
            score = classification_report(Y_test_single_var, Y_pred_single_var)
        else:
             print('Please select a metric from this list:              [f1_score,accuracy_score,recall_score,precision_score,classification_report]')
             sys.exit() 

        scores.append(score)

    return scores



def average_scoring_multiple_metrics(Y_test, Y_pred):

    """Calculates an average score from the four metrics 
       used in the previous function (all apart from 
       classification report).

    Args:
    Y_test: The test dataset for target variables 
            after the train test split 
    Y_pred: The dataset of predicted values for target variables
            after fit and predict
        
    Returns:
    A single score for all 36 variables together which is based on
    the average score of the four metrics
    """
    
    # Average f1 score of 36 target variables
    f1_average=np.mean(average_scoring_single_metric(Y_test, Y_pred,'f1_score'))
    
    # Average accuracy score of 36 target variables    
    accuracy_average=np.mean(average_scoring_single_metric(Y_test, Y_pred,'accuracy_score'))
    
    # Average recall score of 36 target variables
    recall_average=np.mean(average_scoring_single_metric(Y_test, Y_pred,'recall_score'))
    
    # Average precision score of 36 target variables
    precision_average=np.mean(average_scoring_single_metric(Y_test, Y_pred,'precision_score'))

    # Finally, average of the four average scores above
    return (f1_average+accuracy_average+recall_average+precision_average)/4  

def evaluate_model(model, X_test, Y_test, category_names):
         
    """Prints the scores from the two functions above

    Args:
    model: The model used to train,fit and predict data 
    X_test: The test dataset for the X variable
    Y_test: The test dataset for the target variables
    category_names: The names of the target variables
        
    Returns:
    Prints classification report for all 36 variables.
    Then prints the individual score for each of the four metrics
    and the average of these four scores (overall score)
    Finally returns the overall score for tracking purposes
   """
        
    Y_pred=model.predict(X_test)  # Values of target vars predicted
    target_vars=category_names  

    # Classification report for each target variable stored as a list
    report=average_scoring_single_metric(Y_test, Y_pred,'classification_report')

    # Classification report for each target variable printed
    for i in range(len(target_vars)):
        target_var = target_vars[i]
        print(f"Classification Report for {target_var}:")
        print(report[i])
        print()

    
    # Average score of 36 target variables for each of four metrics
    f1_average=np.mean(average_scoring_single_metric(Y_test, Y_pred,'f1_score'))
    accuracy_average=np.mean(average_scoring_single_metric(Y_test, Y_pred,'accuracy_score'))
    recall_average=np.mean(average_scoring_single_metric(Y_test, Y_pred,'recall_score'))
    precision_average=np.mean(average_scoring_single_metric(Y_test, Y_pred,'precision_score'))
    
    # Average of the four individual scores above
    overall_average=average_scoring_multiple_metrics(Y_test, Y_pred)     

    # Print the four individual scores and the overall one 
    print(' Average f1 score: {}\n'.format(f1_average),
          'Average accuracy score: {}\n'.format(accuracy_average),
          'Average recall score: {}\n'.format(recall_average),
          'Average precision score: {}\n\n'.format(precision_average),
          'Overall_average_score: {}'.format(overall_average)
         )
    
    return overall_average

def save_model(model, model_filepath):
    """Saves the final (best) model to a certain path 
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, open(model_filepath, 'wb'))

def build_model_tuned(bootstrap):
    
    """ It builds a tuned model after Grid Search has returned
        the optimum value for argument bootstrap 
    Args:
    bootstrap: The value that Grid Search returned for argmunent
    clf__estimator__bootstrap (True or False)
    #
    This is a very simple case of Grid Search which only tests values
    for clf__estimator__bootstrap. If you check 'parameters' (about 
    80 lines below) you will see that initially there were another 4
    arguments tested. That took over an hour and a half to run so a simpler
    version has been created here. Feel free to uncomment the other 4 
    parameters if you can wait that long. Obvioulsy, you will also
    need to add the corresponding arguments to this function as well.
    
    Returns: The new (tuned) pipeline
    """
    
    pipeline_tuned = Pipeline([
      ('vect', CountVectorizer(tokenizer=tokenize)),  # Add extra parametres if you desire
      ('tfidf', TfidfTransformer()),  # and here
      ('clf',MultiOutputClassifier(RandomForestClassifier(bootstrap=bootstrap))) # and here
    ])
    
    return pipeline_tuned

def build_alt_model():
    
    """ It builds an alternative model using GradientBoostingClassifier
        as the ML method
        
    Args: None needed
    
    Returns: The new pipeline
    """
    
    pipeline_GBC = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(GradientBoostingClassifier()))
    ])
    
    return pipeline_GBC

def main():
    
    """
    The main function that will run the code
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)  # Load data
        
        # Split between train and test dataset
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()  # Initial model
        
        print('Training model...')
        model.fit(X_train, Y_train)  # Train initial model
        
        model_scores={}  # Keep track of scores across various models
        
        print('Evaluating model...')
        # Evaluate initial model and store its score in the dictionary
        model_scores[model]=evaluate_model(model, X_test, 
                                           Y_test, category_names)  
        
        print('Removing variable ''child_alone'' as all its values are 0')
        
        # The evaluation above shows that child_alone is always 0
        # It doesn't need to be predicted as it is stable then
        # Leaving it there might cause an error if an alternative
        # ML method is used
        
        Y_train_modified=Y_train.drop('child_alone',axis=1)
        Y_test_modified=Y_test.drop('child_alone',axis=1)
        category_names_modified=Y_train_modified.columns
        
        print('Building modified model without child alone')
        model_modified = build_model()
 
        print('Training modified model')
        model_modified.fit(X_train, Y_train_modified)
        
        print('Evaluating modified model')
        # Evaluate and store score
        model_scores[model_modified]=evaluate_model(model_modified,
                                                    X_test, 
                                                    Y_test_modified,
                                                    category_names_modified)
        
        print('Getting list of parametres for Grid Searching')
        model_modified.get_params()
        
        # Select parameters for tuned model. As mentioned above
        # feel free to uncomment some (or all) of the parameters
        # below in which case you need to go back up to the 
        # relevant model buidling function and add the 
        # corresponding arguments
        
        parameters = {
            'clf__estimator__bootstrap':[True,False], 
            #'clf__estimator__criterion':['gini','entropy'],
            #'vect__analyzer':['word','char'],
            #'vect__max_features':range(100, 301, 100),
            #'tfidf__smooth_idf':[True,False] 
        }

        # Set the metric by which Grid Search will determine what
        # the optimum params are. This is the function
        # average_scoring_multiple_metrics defined above
        custom_scorer = make_scorer(average_scoring_multiple_metrics)
        
        print('Building Grid Search for the modified model')
        cv = GridSearchCV(model_modified, param_grid=parameters,scoring=custom_scorer)

      
        print('Training Grid Search in order to find the best parametres')
        
        # Time of execution will be counted as this procedure takes long
        # depending on the number of parameters used in the grid
        # Warnings will be hidden here as we have already seen similar
        # messages before (keeps output clear), but will be reset after 
        # the end of this execution
        
        warnings.filterwarnings("ignore")
        start_time = time.time()
        cv.fit(X_train,Y_train_modified)
        end_time = time.time()
        execution_time = end_time - start_time
        warnings.resetwarnings()
        print('Execution time: ',execution_time, ' seconds')
        print('')
        print('Optimum parameters: ',cv.best_params_)
        
        # Setting the optimum parameters for the new model
        # As mentioned above you can uncomment the rest of parameters
        
        bootstrap=cv.best_params_['clf__estimator__bootstrap']
        #criterion=cv.best_params_['clf__estimator__criterion']
        #smooth_idf=cv.best_params_['tfidf__smooth_idf']
        #analyzer=cv.best_params_['vect__analyzer']
        #max_features=cv.best_params_['vect__max_features']

        print('Building the tuned model (using best parametres)')
        model_tuned=build_model_tuned(bootstrap)  # Add the rest of the arguments if needed
        
        print('Training the tuned model')
        model_tuned.fit(X_train,Y_train_modified)
        
        print('Evaluating the tuned model')
        model_scores[model_tuned]=evaluate_model(model_tuned,
                                                 X_test, 
                                                 Y_test_modified,
                                                 category_names_modified)
        
        # GradientBoostingClassifier will be used as an alternative ML method
        print('Building alternative model (GradientBoostingClassifier as the ML method)')
        alt_model=build_alt_model()
        
        print('Training alternative model')
        alt_model.fit(X_train,Y_train_modified)
        
        print('Evaluating alternative model')
        model_scores[alt_model]=evaluate_model(alt_model, X_test,
                                               Y_test_modified,
                                               category_names_modified)
        
        # Only keep the model with the best score
        best_model = max(model_scores, key=model_scores.get)
        print('Score of the best model: ',max(model_scores.values()))
                
        # The pickle file where the model will be saved    
        model_filepath='models/Disaster_model.pkl'  
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        
        save_model(best_model, model_filepath)
        
        print('Trained model saved!')

        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()