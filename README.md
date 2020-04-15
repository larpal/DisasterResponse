# DisasterResponse

##### Table of Contents  
[Project Motivation](#project-Motivation)  
[Data Set](#data-set)  
[Results](#implementation-and-results)  
[File Description](#file-description)  
[Libraries and Dependencies](#libraries-and-Dependencies)  
[Licensing, Authors, and Acknowledgements](#licensing-authors-and-acknowledgements)  

## Project Motivation
This project was done within the framework of the [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) offered by Udacity.
The goals of the project are:
1. To build an ETL pipeline extracting information from a dataset of text messages and loading it into an SQL database
2. To build a machine learning pipeline to predict to which categories each message belongs
3. To build a web application where new messages can be entered and classified and some 
graphics containing information about the data et are displayed.

## Data Set
The data set contains text messages related to natural disasters. The file 
`/data/disaster_messages.csv` contains 26248 text messages in english (partly translated) as 
well as an original language. A second file `/data/disaster_categories.csv` contains binary
labels for 36 categories (such as *floods* or *storms*). Each text message may belong
to multiple categories.

## Implementation and Results
The ETL pipeline consists of the following steps:
* Extract the english messages from the messages data set
* Extract the categories and their labels from the categories data set and create one
column for each category
* Create a table where each row has one text message and 36 binary labels
* Drop duplicate rows
* Drop categories with only one distinct value since we cannot train on a single label.
This applies to the category column `child_alone`
* Store the table in an SQL database

The ML pipeline consists of the following steps:
* Load data from the SQL data base
* Create train and test splits
* Define a machine learning pipeline including:
	* NLP pipeline 
	* Logistic regression classifier for each of the 36 categories
* Train the model using `GridsearchCV` for tuning parameters and cross validation
* Evaluate the model and print out scores.
* Save the model using `pickle`.
The ML pipeline (including training) runs in about two minutes on a 2014 MacBook Pro.

In the web app, the user can manually enter a text message that is then classified for
each of the 36 categories. Further, there are three graphics displayed:
* Distribution of the message genres
* Distribution of the message categories
* Correlations between different categories

*Note: the data set did not include any examples for the category `child_alone`. While 
this category is included in the app, the predictions are not reliable for this category.*

## File Description and Usability
Here is the file structure of the project.
```
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- DisasterResponse.db   # database with cleaned data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py


- models
|- train_classifier.py
|- classifier.pkl  # saved model 


- ETL Pipeline Preparation.ipynb
- ML Pipeline Preparation.ipynb
- README.md
```
The data merged and cleaned by the ETL pipeline is stored in the database `data/DisasterResponse.db`. 
The ETL process can be replicated by running
```python
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```
from within the `data` folder. The ML pipeline includes an NLP processing part and
a logistic regression classifier for each of the 36 message categories. It can be retrained
and replicated by running
```python
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```
from within the `model` folder.

To run the web app locally, run
```python
python run.py
```
from within the `app` folder. The web address to access the app should be displayed in the
terminal.

The two jupyter notebooks `ETL Pipeline Preparation.ipynb` and `ML Pipeline Preparation.ipynb`
have been used to build the ETL and ML pipelines and can be used to experiment with
changes in the pipelines.


## Libraries and Dependencies
The notebook uses the following Python libraries:
* [numpy](https://numpy.org)
* [pandas](https://pandas.pydata.org)
* sqlalchemy
* scikit-laern
* nltk
* sys
* os
* pickle
* json
* plotly
* flask



## Licensing, Authors, and Acknowledgements
The data set was provided by [figure eight](https://www.figure-eight.com) for the 
Udacity Data Science Nano Degree. A similar version of the data set
(which is already preprocessed) is available 
[here](https://www.figure-eight.com/dataset/combined-disaster-response-data/).