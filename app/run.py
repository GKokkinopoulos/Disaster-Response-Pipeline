# import libraries
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)  # initialize flask app

def tokenize(text):

    """Tokenizer function to be used as an argument in the countvectorizer object later

    Args:
    text: A single string value
    
    Returns:
    All tokens in the text as a list
    """   
    
    tokens = word_tokenize(text)  # Split in words-put them in a list
    lemmatizer = WordNetLemmatizer()  # Create an object of WordNetLemmatizer class. 
                                      # Will be used in the loop below

    # The tokens in the list will be lemmatized and added in a new list that will be returned     
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # Lemmatized,lowercase,remove trail space
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')  # create a connection to the database
df = pd.read_sql_table('Disaster_category_messages', engine)  # read the data from the table in the db

# load model
model = joblib.load("../models/Disaster_model.pkl")

def return_graphs():
    """Creates three plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the three plotly visualizations

    """
    
    # first chart plots number of messages by genre as a bar chart
    
    graph_one = []
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # graph created here
    graph_one.append(
       Bar(
          x=genre_names,
          y=genre_counts
          )
    )
    
    # layout for first graph
    layout_one = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre'),
                yaxis = dict(title = 'Count'),
                )
    
    # second chart plots number of messages by category as a bar chart   

    graph_two = []

    # only the 36 category fields remain - sum calculated
    df_cat_count=df.drop(['id','message','original','genre'],axis=1).sum()
    
    # sorted by descending order - list of field names created
    df_cat_count.sort_values(ascending=False,inplace=True)
    df_cat_names=list(df_cat_count.index)
    
    # graph created here
    graph_two.append(
      Bar(
      x = df_cat_names,
      y = df_cat_count,
      )
    )
    
    # layout for graph two
    layout_two = dict(title = 'Messages per category',
                xaxis = dict(title = 'Category'),
                yaxis = dict(title = 'Messages'),
                )

    # third chart plots number of messages by genre and category
    # a line chart for each genre is created
    # the order of categories in x-axis is as in chart two
    
    graph_three = []
    
    # drop fields not needed
    df_genre=df.drop(['id','message','original'],axis=1)
    
    # aggregation takes place here. List "df_cat_names" is used in order
    # to put the fields in the same order as in graph two 
    df_genre_aggr=df_genre.groupby('genre').sum()[df_cat_names].reset_index()
    
    # one line chart is created for each genre and appended to the final graph
    for genre in genre_names:
      x_val = df_cat_names
      
      # squeeze function is used to convert DataFrame (with one row) to Series
      y_val = df_genre_aggr[df_genre_aggr['genre'] == genre].groupby('genre').sum().squeeze()
      
      graph_three.append(
          Scatter(
          x = x_val,
          y = y_val,
          mode = 'lines',
          name = genre
          )
      )
    
    # layout for third chart
    layout_three = dict(title = 'Messages per category and genre',
                xaxis = dict(title = 'Category'),
                yaxis = dict(title = 'Messages'),
                )
      
    # all charts put together - ready to pass on to the html code
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    
    return graphs
    

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # data and layouts for graphs created
    graphs = return_graphs()
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()