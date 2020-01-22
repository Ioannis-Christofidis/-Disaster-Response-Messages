# Disaster Response Pipeline Project

### Purpose of this project:
In the Project, we  retrieve a data set containing real messages that were sent during disaster events. The main purpose of this project is  to create a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier an d saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Results:
The main results of the project can be found running the flask web app

### Acknowledgements:

Must give to the Udacity for having access to the data through it.
