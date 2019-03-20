# Disaster Response Pipeline Project

### Project Summary
This project is part of Data Science Nanodegree program by Udacity, specifically in Data Engineering part, in collaboration with Figure Eight. In this project, we use natural language processing (NLP) technique to train a machine learning model for categorizing disaster events so that messages can be sent to appropriate relief agencies.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterDatabase.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterDatabase.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Folders & Files

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterDatabase.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
